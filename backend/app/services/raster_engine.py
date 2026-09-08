"""
Trinetra AI - Phase 2: Memory-safe Sentinel-2 L2A tiling engine.

Design constraints (16 GB RAM host):
  * Never read a full scene. Every read is a bounded rasterio Window.
  * One chip in flight at a time; the public API is a generator, not a list.
  * Band files stay open for the lifetime of a SceneReader (file handles are
    cheap, re-opening per tile is not), but no band array outlives its chip.
  * GDAL block cache is capped via rasterio.Env so long ingests cannot balloon.

Supported layouts (auto-detected):
  1. Sentinel-2 .SAFE product directory (native Copernicus download)
  2. Directory of per-band GeoTIFFs / JP2s  (B02.tif, B03.tif, ... SCL.tif)
  3. Single stacked multi-band GeoTIFF + an explicit band-index mapping

Resolution handling: B02/B03/B04 are 10 m; B8A/B11/B12 and SCL are 20 m.
All reads are expressed in the 10 m reference grid and each 20 m source is
read with a scaled window + out_shape so every returned band is chip-aligned.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from contextlib import ExitStack
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from app.core.config import settings

logger = logging.getLogger(__name__)

# Native ground sample distance per Sentinel-2 L2A band, in metres.
BAND_RESOLUTION_M: dict[str, int] = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20,
    "SCL": 20,
}
REFERENCE_RESOLUTION_M = 10

SCL_NODATA = 0

_SAFE_DATE_RE = re.compile(r"_(\d{8}T\d{6})_")


class RasterEngineError(RuntimeError):
    """Raised for unrecoverable scene-layout or band-resolution problems."""


# --------------------------------------------------------------------- models
@dataclass(slots=True)
class TileQuality:
    """Per-chip quality metrics, computed from the SCL band before any inference."""

    cloud_fraction: float
    nodata_fraction: float
    stddev: float
    rejected: bool
    reject_reason: str | None = None


@dataclass(slots=True)
class Tile:
    """One spatial chip, carrying both models' band stacks plus geo metadata."""

    tile_id: str
    scene_id: str
    acquisition_timestamp: str | None
    row: int
    col: int
    window: tuple[int, int, int, int]  # col_off, row_off, width, height (10 m grid)
    wgs84_bounding_box: tuple[float, float, float, float]  # W, S, E, N
    crs: str
    quality: TileQuality
    # (3, H, W) uint16 DN, band order B04,B03,B02 - for RemoteCLIP
    rgb: np.ndarray | None = field(default=None, repr=False)
    # (6, H, W) uint16 DN, band order = settings.prithvi_bands (B02,B03,B04,B8A,B11,B12) - for Prithvi
    prithvi_stack: np.ndarray | None = field(default=None, repr=False)

    def payload(self) -> dict:
        """Qdrant payload - geometry and quality only, never pixel arrays."""
        return {
            "tile_id": self.tile_id,
            "scene_id": self.scene_id,
            "acquisition_timestamp": self.acquisition_timestamp,
            "wgs84_bounding_box": list(self.wgs84_bounding_box),
            "scl_cloud_coverage": round(self.quality.cloud_fraction, 4),
            "nodata_fraction": round(self.quality.nodata_fraction, 4),
            "crs": self.crs,
            "row": self.row,
            "col": self.col,
            "window": list(self.window),
        }

    def release(self) -> None:
        """Drop pixel references so the arrays are collectable immediately."""
        self.rgb = None
        self.prithvi_stack = None


# ------------------------------------------------------------- band discovery
def _index_safe_product(root: Path) -> dict[str, Path]:
    """Map band name -> JP2 path inside a .SAFE L2A product."""
    granule = root / "GRANULE"
    if not granule.is_dir():
        raise RasterEngineError(f"{root} has no GRANULE/ - not a .SAFE product.")

    bands: dict[str, Path] = {}
    # IMG_DATA/R10m/T43RGN_20240115T052131_B02_10m.jp2
    for img in granule.rglob("IMG_DATA/**/*.jp2"):
        m = re.search(r"_(B[0-9A]{2}|SCL)_(\d{2})m\.jp2$", img.name)
        if not m:
            continue
        band, res = m.group(1), int(m.group(2))
        native = BAND_RESOLUTION_M.get(band)
        # Prefer the native-resolution rendition; ignore resampled duplicates.
        if native is not None and res == native:
            bands[band] = img
        elif band not in bands:
            bands[band] = img
    if not bands:
        raise RasterEngineError(f"No band JP2s found under {granule}")
    return bands


def _index_band_directory(root: Path) -> dict[str, Path]:
    """Map band name -> path for a flat directory of per-band rasters."""
    bands: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in {".tif", ".tiff", ".jp2"}:
            continue
        m = re.search(r"(B[0-9A]{2}|SCL)", path.stem, re.IGNORECASE)
        if m:
            bands.setdefault(m.group(1).upper(), path)
    return bands


def _parse_safe_timestamp(name: str) -> str | None:
    m = _SAFE_DATE_RE.search(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------- scene reader
class SceneReader:
    """
    Windowed reader over one Sentinel-2 L2A scene.

    Use as a context manager so every dataset handle is closed deterministically:

        with SceneReader(path) as scene:
            for tile in scene.iter_tiles():
                ...
    """

    def __init__(
        self,
        source: str | Path,
        *,
        scene_id: str | None = None,
        acquisition_timestamp: str | None = None,
        stacked_band_map: dict[str, int] | None = None,
    ) -> None:
        self.source = Path(source)
        if not self.source.exists():
            raise RasterEngineError(f"Scene not found: {self.source}")

        self.scene_id = scene_id or self.source.stem
        self.acquisition_timestamp = (
            acquisition_timestamp or _parse_safe_timestamp(self.source.name)
        )
        self._stack: ExitStack | None = None
        self._datasets: dict[str, rasterio.DatasetReader] = {}
        self._stacked_band_map = stacked_band_map
        self._stacked_ds: rasterio.DatasetReader | None = None

        self.required_bands: list[str] = list(
            dict.fromkeys(settings.prithvi_bands + settings.remoteclip_bands)
        )

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> "SceneReader":
        self._stack = ExitStack()
        env = self._stack.enter_context(
            rasterio.Env(
                GDAL_CACHEMAX=settings.gdal_cachemax_mb,
                GDAL_NUM_THREADS=settings.gdal_num_threads,
                # COGs only; stops GDAL probing sidecar files on every open.
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                VSI_CACHE="FALSE",
            )
        )
        del env
        try:
            self._open_sources()
        except Exception:
            self._stack.close()
            self._stack = None
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self._datasets.clear()
        self._stacked_ds = None
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        gc.collect()

    def _open_sources(self) -> None:
        assert self._stack is not None

        if self.source.is_file():
            self._open_stacked()
        elif (self.source / "GRANULE").is_dir() or self.source.suffix.upper() == ".SAFE":
            self._open_multi(_index_safe_product(self.source))
        elif self.source.is_dir():
            found = _index_band_directory(self.source)
            if not found:
                raise RasterEngineError(
                    f"{self.source} contains no recognizable Sentinel-2 band rasters "
                    "(expected names like B02..B12 / B8A / SCL)."
                )
            self._open_multi(found)
        else:
            raise RasterEngineError(f"Unsupported scene source: {self.source}")

        self._validate_geometry()

    def _open_multi(self, band_paths: dict[str, Path]) -> None:
        assert self._stack is not None
        missing = [b for b in self.required_bands if b not in band_paths]
        if missing:
            raise RasterEngineError(
                f"Scene {self.scene_id} missing required band(s): {missing}. "
                f"Found: {sorted(band_paths)}"
            )
        if "SCL" not in band_paths:
            logger.warning(
                "Scene %s has no SCL band - cloud masking disabled, all tiles pass "
                "the cloud test. Supply an L2A product with SCL for real filtering.",
                self.scene_id,
            )

        wanted = list(self.required_bands) + (["SCL"] if "SCL" in band_paths else [])
        for band in wanted:
            path = band_paths[band]
            try:
                self._datasets[band] = self._stack.enter_context(rasterio.open(path))
            except RasterioIOError as exc:
                raise RasterEngineError(f"Cannot open {band} at {path}: {exc}") from exc

        if self.acquisition_timestamp is None:
            for path in band_paths.values():
                ts = _parse_safe_timestamp(path.name)
                if ts:
                    self.acquisition_timestamp = ts
                    break

    def _open_stacked(self) -> None:
        """Single multi-band GeoTIFF. Band mapping comes from an explicit dict or a sidecar."""
        assert self._stack is not None
        try:
            ds = self._stack.enter_context(rasterio.open(self.source))
        except RasterioIOError as exc:
            raise RasterEngineError(f"Cannot open {self.source}: {exc}") from exc
        self._stacked_ds = ds

        band_map = self._stacked_band_map
        if band_map is None:
            sidecar = self.source.with_suffix(".bands.json")
            if sidecar.is_file():
                band_map = {k.upper(): int(v) for k, v in
                            json.loads(sidecar.read_text()).items()}
            elif ds.count >= 6:
                # Assume the canonical Prithvi order in bands 1..6.
                band_map = {b: i + 1 for i, b in enumerate(settings.prithvi_bands)}
                logger.warning(
                    "No band map for %s; assuming bands 1-6 are %s. Provide a "
                    "%s sidecar to be explicit.",
                    self.source.name, settings.prithvi_bands, sidecar.name,
                )
            else:
                raise RasterEngineError(
                    f"{self.source} has {ds.count} band(s); need >=6 or an explicit "
                    "stacked_band_map."
                )

        missing = [b for b in self.required_bands if b not in band_map]
        if missing:
            raise RasterEngineError(f"stacked_band_map missing band(s): {missing}")
        bad = {b: i for b, i in band_map.items() if not 1 <= i <= ds.count}
        if bad:
            raise RasterEngineError(
                f"stacked_band_map indices out of range for {ds.count}-band file: {bad}"
            )
        self._stacked_band_map = band_map

    def _validate_geometry(self) -> None:
        """All sources must share a CRS; resolutions must be integer multiples of 10 m."""
        ref = self._reference_dataset()
        self.crs = ref.crs
        if self.crs is None:
            raise RasterEngineError(f"Scene {self.scene_id} has no CRS.")
        self.transform = ref.transform
        self.height, self.width = ref.height, ref.width

        for band, ds in self._datasets.items():
            if ds.crs != self.crs:
                raise RasterEngineError(
                    f"CRS mismatch: {band} is {ds.crs}, reference is {self.crs}."
                )
            ratio = ds.transform[0] / self.transform[0]
            if not np.isclose(ratio, round(ratio), atol=1e-3):
                raise RasterEngineError(
                    f"{band} resolution {ds.transform[0]} is not an integer multiple "
                    f"of the reference 10 m grid resolution {self.transform[0]}."
                )

    def _reference_dataset(self) -> rasterio.DatasetReader:
        """The 10 m grid that all windows are expressed in."""
        if self._stacked_ds is not None:
            return self._stacked_ds
        for band in ("B02", "B03", "B04"):
            if band in self._datasets:
                return self._datasets[band]
        return next(iter(self._datasets.values()))

    @property
    def has_scl(self) -> bool:
        return "SCL" in self._datasets

    # ----------------------------------------------------------------- reads
    def _read_band(self, band: str, window: Window, out_shape: tuple[int, int]) -> np.ndarray:
        """
        Read one band over a 10 m-grid window, resampled to out_shape.

        20 m bands get a scaled window so the same ground footprint is read from
        fewer source pixels, then nearest/bilinear-resampled up to the chip size.
        """
        if self._stacked_ds is not None:
            ds = self._stacked_ds
            index = self._stacked_band_map[band]  # type: ignore[index]
            src_window = window
        else:
            ds = self._datasets[band]
            index = 1
            scale = round(ds.transform[0] / self.transform[0])  # 1 for 10 m, 2 for 20 m
            src_window = Window(
                col_off=window.col_off / scale,
                row_off=window.row_off / scale,
                width=window.width / scale,
                height=window.height / scale,
            )

        # SCL is categorical - nearest only. Reflectance can be bilinear.
        resampling = Resampling.nearest if band == "SCL" else Resampling.bilinear

        arr = ds.read(
            index,
            window=src_window,
            out_shape=out_shape,
            resampling=resampling,
            boundless=True,
            fill_value=0,
        )
        return arr

    def _read_scl(self, window: Window, out_shape: tuple[int, int]) -> np.ndarray | None:
        if not self.has_scl:
            return None
        return self._read_band("SCL", window, out_shape).astype(np.uint8, copy=False)

    # ------------------------------------------------------------- quality
    @staticmethod
    def _assess(
        scl: np.ndarray | None,
        reference: np.ndarray,
    ) -> TileQuality:
        """
        Cloud/no-data/variance gate. Runs before any GPU work so rejected chips
        cost nothing but one windowed read.
        """
        total = reference.shape[-2] * reference.shape[-1]

        if scl is not None:
            nodata_px = int(np.count_nonzero(scl == SCL_NODATA))
            reject_mask = np.isin(scl, settings.scl_reject_classes)
            cloud_px = int(np.count_nonzero(reject_mask))
            del reject_mask
        else:
            # No SCL: fall back to zero-valued reflectance as the no-data proxy.
            nodata_px = int(np.count_nonzero(reference[0] == 0))
            cloud_px = 0

        nodata_fraction = nodata_px / total
        valid = total - nodata_px
        cloud_fraction = (cloud_px / valid) if valid > 0 else 1.0

        # Variance on valid pixels only, in float32 to avoid uint16 overflow.
        sample = reference[0]
        stddev = float(np.std(sample[sample > 0].astype(np.float32))) if valid else 0.0

        if nodata_fraction > settings.max_nodata_fraction:
            return TileQuality(cloud_fraction, nodata_fraction, stddev, True,
                               f"nodata {nodata_fraction:.1%} > "
                               f"{settings.max_nodata_fraction:.0%}")
        if cloud_fraction > settings.max_cloud_fraction:
            return TileQuality(cloud_fraction, nodata_fraction, stddev, True,
                               f"cloud/cirrus {cloud_fraction:.1%} > "
                               f"{settings.max_cloud_fraction:.0%}")
        if stddev < settings.min_stddev:
            return TileQuality(cloud_fraction, nodata_fraction, stddev, True,
                               f"flat chip (stddev {stddev:.2f})")
        return TileQuality(cloud_fraction, nodata_fraction, stddev, False)

    # --------------------------------------------------------------- geometry
    def window_bounds_wgs84(self, window: Window) -> tuple[float, float, float, float]:
        left, bottom, right, top = rasterio.windows.bounds(window, self.transform)
        w, s, e, n = transform_bounds(
            self.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
        )
        return (round(w, 7), round(s, 7), round(e, 7), round(n, 7))

    def grid_shape(self) -> tuple[int, int]:
        stride = settings.tile_stride
        rows = max(0, (self.height - settings.tile_size) // stride + 1)
        cols = max(0, (self.width - settings.tile_size) // stride + 1)
        return rows, cols

    def tile_window(self, row: int, col: int) -> Window:
        """The 10 m-grid window for one grid cell. Raises if out of bounds."""
        rows, cols = self.grid_shape()
        if not (0 <= row < rows and 0 <= col < cols):
            raise RasterEngineError(
                f"tile r{row}c{col} outside grid {rows}x{cols} for {self.scene_id}"
            )
        stride = settings.tile_stride
        return Window(col * stride, row * stride, settings.tile_size, settings.tile_size)

    # --------------------------------------------------- random-access reads
    def read_tile_stack(
        self,
        row: int,
        col: int,
        bands: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        Read one grid cell's bands as (N, H, W) uint16 without streaming.

        Used by the change-detection path, which needs the SAME window from a
        second scene rather than a sequential sweep.
        """
        window = self.tile_window(row, col)
        size = settings.tile_size
        return self._stack_bands(bands or settings.prithvi_bands, window, (size, size))

    def read_tile_rgb8(self, row: int, col: int) -> np.ndarray:
        """One grid cell as (H, W, 3) uint8, stretched - for on-demand previews."""
        return to_rgb_uint8(self.read_tile_stack(row, col, settings.remoteclip_bands))

    def tile_quality(self, row: int, col: int) -> TileQuality:
        """Run the cloud/no-data/variance gate on a single grid cell."""
        window = self.tile_window(row, col)
        size = settings.tile_size
        out_shape = (size, size)
        probe = self._read_band("B04", window, out_shape)
        scl = self._read_scl(window, out_shape)
        try:
            return self._assess(scl, probe[np.newaxis, ...])
        finally:
            probe = None
            scl = None

    def is_grid_compatible(self, other: "SceneReader") -> str | None:
        """
        Returns None when two scenes share a tiling grid, else a reason string.

        Change detection compares chip r,c in T1 against chip r,c in T2, so a
        CRS or transform mismatch would silently compare different ground.
        """
        if self.crs != other.crs:
            return f"CRS mismatch: {self.crs} vs {other.crs}"
        if (self.width, self.height) != (other.width, other.height):
            return (f"raster size mismatch: {self.width}x{self.height} vs "
                    f"{other.width}x{other.height}")
        a, b = self.transform, other.transform
        if not (np.isclose(a.c, b.c, atol=1.0) and np.isclose(a.f, b.f, atol=1.0)
                and np.isclose(a.a, b.a) and np.isclose(a.e, b.e)):
            return "geotransform origin/pixel size mismatch (scenes not co-registered)"
        return None

    # ------------------------------------------------------------ tile stream
    def iter_tiles(
        self,
        *,
        load_pixels: bool = True,
        yield_rejected: bool = False,
        max_tiles: int | None = None,
        auto_release: bool = True,
    ) -> Iterator[Tile]:
        """
        Stream chips across the scene, left-to-right, top-to-bottom.

        Only one chip's arrays exist at a time.

        auto_release=True (default) frees each chip's pixel arrays as soon as the
        consumer resumes the generator - correct for the streaming ingest path.
        Set auto_release=False if you intend to collect tiles into a list, but be
        aware that retains every chip's arrays in RAM.
        """
        rows, cols = self.grid_shape()
        if rows == 0 or cols == 0:
            logger.warning(
                "Scene %s (%dx%d) is smaller than one %dpx tile - nothing to do.",
                self.scene_id, self.width, self.height, settings.tile_size,
            )
            return

        stride = settings.tile_stride
        size = settings.tile_size
        out_shape = (size, size)
        emitted = 0
        stats = {"total": 0, "rejected": 0}

        for row in range(rows):
            for col in range(cols):
                if max_tiles is not None and emitted >= max_tiles:
                    logger.info("Stopped at max_tiles=%d", max_tiles)
                    self._log_summary(stats)
                    return

                stats["total"] += 1
                window = Window(col * stride, row * stride, size, size)

                # Cheapest sufficient read first: B04 drives the variance test,
                # SCL drives cloud/no-data. Both are dropped before any band stack
                # is allocated, so a rejected chip never holds two arrays at once.
                scl = None
                try:
                    probe = self._read_band("B04", window, out_shape)
                    scl = self._read_scl(window, out_shape)
                    quality = self._assess(scl, probe[np.newaxis, ...])
                except Exception as exc:
                    logger.error("Quality probe failed at r%dc%d: %s", row, col, exc)
                    scl = None
                    continue
                finally:
                    probe = None

                # SCL is no longer needed once quality is decided. Rebind rather
                # than `del` so the fall-through path below stays idempotent when
                # yield_rejected=True.
                scl = None

                if quality.rejected:
                    stats["rejected"] += 1
                    logger.debug("skip r%dc%d: %s", row, col, quality.reject_reason)
                    if not yield_rejected:
                        continue

                tile = Tile(
                    tile_id=f"{self.scene_id}_r{row:04d}c{col:04d}",
                    scene_id=self.scene_id,
                    acquisition_timestamp=self.acquisition_timestamp,
                    row=row,
                    col=col,
                    window=(int(window.col_off), int(window.row_off),
                            int(window.width), int(window.height)),
                    wgs84_bounding_box=self.window_bounds_wgs84(window),
                    crs=str(self.crs),
                    quality=quality,
                )

                if load_pixels and not quality.rejected:
                    try:
                        tile.rgb = self._stack_bands(settings.remoteclip_bands,
                                                     window, out_shape)
                        tile.prithvi_stack = self._stack_bands(settings.prithvi_bands,
                                                               window, out_shape)
                    except Exception as exc:
                        logger.error("Read failed for %s: %s", tile.tile_id, exc)
                        tile.release()
                        continue

                emitted += 1
                yield tile

                # The consumer is done with this chip; make it collectable now
                # rather than at the next generator resume.
                if auto_release:
                    tile.release()
                if emitted % 256 == 0:
                    gc.collect()

        self._log_summary(stats)

    def _stack_bands(
        self,
        bands: Sequence[str],
        window: Window,
        out_shape: tuple[int, int],
    ) -> np.ndarray:
        """Read bands into one (N, H, W) uint16 array, preallocated to avoid copies."""
        out = np.empty((len(bands), *out_shape), dtype=np.uint16)
        hi = np.iinfo(np.uint16).max
        for i, band in enumerate(bands):
            arr = self._read_band(band, window, out_shape)
            if arr.dtype == np.uint16:
                out[i] = arr
            else:
                # Bilinear resampling can emit float/int16; clamp before narrowing.
                out[i] = np.clip(arr, 0, hi).astype(np.uint16)
            del arr
        return out

    @staticmethod
    def _log_summary(stats: dict[str, int]) -> None:
        total, rejected = stats["total"], stats["rejected"]
        if total:
            logger.info(
                "Tiling complete: %d candidate(s), %d rejected (%.1f%%), %d kept.",
                total, rejected, 100.0 * rejected / total, total - rejected,
            )


# ----------------------------------------------------------------- utilities
def to_rgb_uint8(rgb_dn: np.ndarray, percentile: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """
    (3, H, W) uint16 DN -> (H, W, 3) uint8 for PNG preview and RemoteCLIP input.

    Percentile stretch per band: Sentinel-2 L2A surface reflectance is dark and
    low-contrast without it, which measurably degrades CLIP embeddings.
    """
    if rgb_dn.ndim != 3 or rgb_dn.shape[0] != 3:
        raise ValueError(f"expected (3, H, W), got {rgb_dn.shape}")

    out = np.empty((rgb_dn.shape[1], rgb_dn.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_dn[i].astype(np.float32)
        valid = band[band > 0]
        if valid.size == 0:
            out[..., i] = 0
            continue
        lo, hi = np.percentile(valid, percentile)
        if hi <= lo:
            hi = lo + 1.0
        np.clip((band - lo) * (255.0 / (hi - lo)), 0, 255, out=band)
        out[..., i] = band.astype(np.uint8)
        del band, valid
    return out


def normalize_for_prithvi(stack_dn: np.ndarray) -> np.ndarray:
    """
    (6, H, W) uint16 DN -> (6, H, W) float32 standardized with Prithvi's own
    channel statistics from its config.json. Band order must already be
    B02,B03,B04,B8A,B11,B12.
    """
    expected = len(settings.prithvi_bands)
    if stack_dn.ndim != 3 or stack_dn.shape[0] != expected:
        raise ValueError(f"expected ({expected}, H, W), got {stack_dn.shape}")

    mean = np.asarray(settings.prithvi_mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(settings.prithvi_std, dtype=np.float32).reshape(-1, 1, 1)
    out = stack_dn.astype(np.float32)
    out -= mean
    out /= std
    return out
