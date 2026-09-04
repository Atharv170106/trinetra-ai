"""
Trinetra AI - Phase 2 verification.

Builds a synthetic Sentinel-2 L2A scene on disk (correct CRS, correct 10 m/20 m
resolution split, a real SCL band with planted cloud and no-data regions), then
exercises the tiling engine against it.

Run from the repo root:  python backend/test_raster_engine.py

Checks:
  1. Multi-resolution reads stay chip-aligned (20 m bands upsampled correctly)
  2. SCL cloud rejection fires at the >20% threshold
  3. No-data (scene edge) rejection fires
  4. Flat-chip variance rejection fires
  5. WGS84 bounding boxes are sane and monotonic across the grid
  6. Band ordering is preserved for both models
  7. RSS memory stays flat while streaming the whole scene
  8. Stacked single-GeoTIFF layout works via an explicit band map
  9. Missing-band and CRS-mismatch failures raise RasterEngineError
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings  # noqa: E402
from app.services.raster_engine import (  # noqa: E402
    RasterEngineError,
    SceneReader,
    normalize_for_prithvi,
    to_rgb_uint8,
)

PASS, FAIL = "[ PASS ]", "[ FAIL ]"
_failures: list[str] = []

# 10 m grid: 1024x1024 -> 4x4 grid of 256px tiles.
SCENE_W = SCENE_H = 1024
CRS = "EPSG:32643"          # UTM 43N, covers northern India
ORIGIN_X, ORIGIN_Y = 300000.0, 3200000.0

ALL_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07")


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def bad(check: str, msg: str) -> None:
    print(f"{FAIL} {check}: {msg}")
    _failures.append(check)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# ------------------------------------------------------------ scene synthesis
def _write(path: Path, arr: np.ndarray, res: int, crs: str = CRS) -> None:
    transform = from_origin(ORIGIN_X, ORIGIN_Y, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs, transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def build_scene(root: Path) -> None:
    """
    Per-band GeoTIFFs with a planted layout on the 4x4 tile grid:
      row 0        -> clean, textured
      row 1, col 0 -> 100% cloud (SCL 9)      -> reject
      row 1, col 1 -> 50% thin cirrus (SCL 10)-> reject
      row 1, col 2 -> 10% cloud               -> keep
      row 2        -> no-data (SCL 0)         -> reject
      row 3, col 0 -> flat (constant value)   -> reject
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    size = settings.tile_size  # 256

    for i, band in enumerate(ALL_BANDS):
        res = 10 if band in ("B02", "B03", "B04") else 20
        dim = SCENE_W if res == 10 else SCENE_W // 2
        # Textured base so the variance gate passes on clean chips.
        arr = (rng.integers(400, 3000, size=(dim, dim)) + i * 120).astype(np.uint16)

        s = size if res == 10 else size // 2
        arr[2 * s:3 * s, :] = 0          # row 2 -> no-data
        arr[3 * s:4 * s, 0:s] = 1500     # row 3 col 0 -> flat
        _write(root / f"{band}.tif", arr, res)
        del arr

    # SCL at 20 m -> 128px per 256px tile.
    s = size // 2
    scl = np.full((SCENE_W // 2, SCENE_H // 2), 4, dtype=np.uint8)  # 4 = vegetation
    scl[1 * s:2 * s, 0 * s:1 * s] = 9                    # full cloud high prob
    scl[1 * s:2 * s, 1 * s:1 * s + s // 2] = 10          # half thin cirrus
    scl[1 * s:1 * s + s // 10, 2 * s:3 * s] = 9          # ~10% cloud -> keep
    scl[2 * s:3 * s, :] = 0                              # no-data row
    _write(root / "SCL.tif", scl, 20)
    del scl


def build_stacked(path: Path) -> dict[str, int]:
    """Single 6-band 10 m GeoTIFF + sidecar band map."""
    rng = np.random.default_rng(7)
    stack = (rng.integers(400, 3000, size=(6, SCENE_W, SCENE_H))).astype(np.uint16)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=SCENE_H, width=SCENE_W, count=6,
        dtype="uint16", crs=CRS, transform=transform, tiled=True,
        blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(stack)
    band_map = {b: i + 1 for i, b in enumerate(settings.prithvi_bands)}
    path.with_suffix(".bands.json").write_text(json.dumps(band_map))
    del stack
    return band_map


# ----------------------------------------------------------------- the checks
def check_grid_and_geometry(scene_dir: Path) -> None:
    section("Grid & geometry")
    with SceneReader(scene_dir, scene_id="SYNTH_T43RGN") as scene:
        rows, cols = scene.grid_shape()
        if (rows, cols) == (4, 4):
            ok(f"grid {rows}x{cols} over {scene.width}x{scene.height} @ "
               f"{settings.tile_size}px")
        else:
            bad("grid_shape", f"expected 4x4, got {rows}x{cols}")

        if scene.has_scl:
            ok("SCL band detected")
        else:
            bad("SCL detection", "SCL.tif present but not picked up")

        first = next(scene.iter_tiles(load_pixels=False, yield_rejected=True))
        w, s, e, n = first.wgs84_bounding_box
        # UTM 43N at this northing -> roughly 76-78E, 28-29N.
        if 70 < w < 82 and 24 < s < 32 and e > w and n > s:
            ok(f"WGS84 bbox plausible: W={w:.5f} S={s:.5f} E={e:.5f} N={n:.5f}")
        else:
            bad("bbox", f"implausible WGS84 bounds {first.wgs84_bounding_box}")

        # Longitude must increase left-to-right within row 0.
        row0 = [t for t in scene.iter_tiles(load_pixels=False, yield_rejected=True)
                if t.row == 0]
        lons = [t.wgs84_bounding_box[0] for t in row0]
        if lons == sorted(lons) and len(set(lons)) == len(lons):
            ok(f"longitude strictly increases across row 0 ({len(lons)} tiles)")
        else:
            bad("bbox ordering", f"non-monotonic longitudes: {lons}")


def check_quality_filters(scene_dir: Path) -> None:
    section("SCL / quality filtering")
    with SceneReader(scene_dir, scene_id="SYNTH") as scene:
        by_rc = {
            (t.row, t.col): t
            for t in scene.iter_tiles(load_pixels=False, yield_rejected=True)
        }

    expectations = {
        (0, 0): (False, "clean"),
        (1, 0): (True, "100% cloud high-prob"),
        (1, 1): (True, "50% thin cirrus"),
        (1, 2): (False, "~10% cloud, under threshold"),
        (2, 0): (True, "no-data row"),
        (3, 0): (True, "flat chip"),
    }
    for (r, c), (want_reject, label) in expectations.items():
        tile = by_rc.get((r, c))
        if tile is None:
            bad(f"tile r{r}c{c}", "not produced")
            continue
        got = tile.quality.rejected
        detail = (f"cloud={tile.quality.cloud_fraction:.1%} "
                  f"nodata={tile.quality.nodata_fraction:.1%} "
                  f"std={tile.quality.stddev:.1f}")
        if got == want_reject:
            verdict = f"rejected ({tile.quality.reject_reason})" if got else "kept"
            ok(f"r{r}c{c} {label}: {verdict} | {detail}")
        else:
            bad(f"filter r{r}c{c}",
                f"{label}: expected rejected={want_reject}, got {got} | {detail}")

    kept = sum(1 for t in by_rc.values() if not t.quality.rejected)
    ok(f"{kept}/{len(by_rc)} tiles survive filtering")


def check_band_stacks(scene_dir: Path) -> None:
    section("Band stacks & model inputs")
    with SceneReader(scene_dir, scene_id="SYNTH") as scene:
        tile = next(scene.iter_tiles(load_pixels=True))

        size = settings.tile_size
        if tile.rgb is not None and tile.rgb.shape == (3, size, size):
            ok(f"RGB stack {tile.rgb.shape} dtype={tile.rgb.dtype}")
        else:
            bad("rgb shape", f"got {None if tile.rgb is None else tile.rgb.shape}")
            return

        if tile.prithvi_stack is not None and tile.prithvi_stack.shape == (6, size, size):
            ok(f"Prithvi stack {tile.prithvi_stack.shape} dtype={tile.prithvi_stack.dtype}")
        else:
            bad("prithvi shape",
                f"got {None if tile.prithvi_stack is None else tile.prithvi_stack.shape}")
            return

        # Synthetic bands are offset by i*120, so ordering is verifiable by mean.
        means = [float(tile.prithvi_stack[i].mean()) for i in range(6)]
        if all(means[i] < means[i + 1] for i in range(5)):
            ok(f"band order B02..B07 preserved (means {[round(m) for m in means]})")
        else:
            bad("band order", f"means not ascending: {[round(m) for m in means]}")

        # 20 m bands must not come back as a half-size array or a constant block.
        b05 = tile.prithvi_stack[3]
        if b05.shape == (size, size) and float(b05.std()) > 1.0:
            ok(f"20 m band B05 upsampled to {b05.shape}, std={b05.std():.1f}")
        else:
            bad("20m upsample", f"B05 shape={b05.shape} std={b05.std():.2f}")

        rgb8 = to_rgb_uint8(tile.rgb)
        if rgb8.shape == (size, size, 3) and rgb8.dtype == np.uint8 and rgb8.max() > 200:
            ok(f"to_rgb_uint8 -> {rgb8.shape} range [{rgb8.min()}, {rgb8.max()}]")
        else:
            bad("to_rgb_uint8", f"shape={rgb8.shape} dtype={rgb8.dtype} max={rgb8.max()}")

        norm = normalize_for_prithvi(tile.prithvi_stack)
        if norm.shape == (6, size, size) and norm.dtype == np.float32 and abs(norm.mean()) < 5:
            ok(f"normalize_for_prithvi -> {norm.dtype} mean={norm.mean():.3f} "
               f"std={norm.std():.3f}")
        else:
            bad("normalize_for_prithvi",
                f"shape={norm.shape} dtype={norm.dtype} mean={norm.mean():.3f}")

        payload = tile.payload()
        required = {"tile_id", "scene_id", "acquisition_timestamp",
                    "wgs84_bounding_box", "scl_cloud_coverage"}
        missing = required - payload.keys()
        if missing:
            bad("payload", f"missing keys {missing}")
        elif any(isinstance(v, np.ndarray) for v in payload.values()):
            bad("payload", "contains a numpy array - not JSON serializable")
        else:
            json.dumps(payload)  # raises if not serializable
            ok(f"payload JSON-safe: {payload['tile_id']} "
               f"cloud={payload['scl_cloud_coverage']}")


def check_memory_stability(scene_dir: Path) -> None:
    section("Memory stability (streaming full scene)")
    try:
        import psutil
    except ImportError:
        print("        psutil unavailable - skipping RSS measurement")
        return

    proc = psutil.Process()
    baseline = proc.memory_info().rss / 1024**2

    peak = baseline
    count = 0
    with SceneReader(scene_dir, scene_id="SYNTH") as scene:
        for tile in scene.iter_tiles(load_pixels=True, yield_rejected=True):
            count += 1
            peak = max(peak, proc.memory_info().rss / 1024**2)
            del tile

    after = proc.memory_info().rss / 1024**2
    growth = after - baseline
    print(f"        baseline={baseline:.1f} MB peak={peak:.1f} MB "
          f"after={after:.1f} MB over {count} tiles")
    if growth < 150:
        ok(f"RSS growth {growth:+.1f} MB - no accumulation")
    else:
        bad("memory", f"RSS grew {growth:+.1f} MB streaming {count} tiles")


def check_stacked_layout(tmp: Path) -> None:
    section("Stacked single-GeoTIFF layout")
    stacked = tmp / "stacked_scene.tif"
    build_stacked(stacked)
    try:
        with SceneReader(stacked, scene_id="STACKED") as scene:
            if not scene.has_scl:
                ok("no SCL present -> cloud masking correctly disabled (warned)")
            tile = next(scene.iter_tiles(load_pixels=True))
            if (tile.prithvi_stack is not None
                    and tile.prithvi_stack.shape == (6, settings.tile_size, settings.tile_size)):
                ok(f"stacked read via sidecar band map -> {tile.prithvi_stack.shape}")
            else:
                bad("stacked read", "unexpected stack shape")
    except Exception as exc:
        bad("stacked layout", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def check_error_handling(tmp: Path) -> None:
    section("Error handling")
    # Missing band
    partial = tmp / "partial"
    partial.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for band in ("B02", "B03"):
        _write(partial / f"{band}.tif",
               rng.integers(0, 3000, (256, 256)).astype(np.uint16), 10)
    try:
        with SceneReader(partial):
            bad("missing band", "should have raised RasterEngineError")
    except RasterEngineError as exc:
        ok(f"missing bands rejected: {str(exc)[:70]}...")
    except Exception as exc:
        bad("missing band", f"wrong exception {type(exc).__name__}: {exc}")

    # CRS mismatch
    mixed = tmp / "mixed_crs"
    mixed.mkdir(exist_ok=True)
    for band in ALL_BANDS:
        res = 10 if band in ("B02", "B03", "B04") else 20
        dim = 512 if res == 10 else 256
        crs = "EPSG:32644" if band == "B07" else CRS
        _write(mixed / f"{band}.tif",
               rng.integers(0, 3000, (dim, dim)).astype(np.uint16), res, crs=crs)
    try:
        with SceneReader(mixed):
            bad("crs mismatch", "should have raised RasterEngineError")
    except RasterEngineError as exc:
        ok(f"CRS mismatch rejected: {str(exc)[:70]}...")
    except Exception as exc:
        bad("crs mismatch", f"wrong exception {type(exc).__name__}: {exc}")

    # Nonexistent path
    try:
        SceneReader(tmp / "does_not_exist")
        bad("missing path", "should have raised RasterEngineError")
    except RasterEngineError:
        ok("nonexistent scene path rejected")


# -------------------------------------------------------------------- runner
def main() -> int:
    print("=" * 68)
    print(" TRINETRA AI - PHASE 2 RASTER ENGINE VERIFICATION")
    print("=" * 68)
    print(f"tile_size={settings.tile_size} stride={settings.tile_stride} "
          f"max_cloud={settings.max_cloud_fraction:.0%} "
          f"reject_classes={settings.scl_reject_classes}")

    tmp = Path(tempfile.mkdtemp(prefix="trinetra_phase2_"))
    try:
        scene_dir = tmp / "synthetic_scene"
        print(f"\nbuilding synthetic L2A scene in {scene_dir} ...")
        build_scene(scene_dir)
        ok("synthetic scene written (10 m B02-B04, 20 m B05-B07 + SCL)")

        for check in (check_grid_and_geometry, check_quality_filters,
                      check_band_stacks, check_memory_stability):
            try:
                check(scene_dir)
            except Exception as exc:
                bad(check.__name__, f"unhandled {type(exc).__name__}: {exc}")
                traceback.print_exc()

        for check in (check_stacked_layout, check_error_handling):
            try:
                check(tmp)
            except Exception as exc:
                bad(check.__name__, f"unhandled {type(exc).__name__}: {exc}")
                traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    section("Summary")
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed: {', '.join(_failures)}")
    else:
        print(f"{PASS} Phase 2 raster engine verified.")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
