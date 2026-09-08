#!/usr/bin/env python3
"""
Trinetra AI - Secure Ingestion Pipeline (STAGING TOOL, ONLINE ONLY)
===================================================================

Fetches Sentinel-2 L2A COG bands from a public STAC catalogue, crops them to an
AOI with windowed HTTP range reads, and writes them into the secure drop zone
for the air-gapped watchdog to pick up.

  !! THIS SCRIPT REQUIRES INTERNET AND IS NOT PART OF THE DEPLOYED STACK. !!

It is deliberately excluded from the Docker image (.dockerignore). The intended
operational model is sneakernet: run this on a connected staging machine, move
the resulting directory onto the air-gapped host, drop it in secure_drop_zone/.

Output layout (SceneReader "Layout 2" - a directory of per-band GeoTIFFs):

    backend/secure_drop_zone/
        S2A_T43RGN_20250115T052131_L2A/
            B02.tif  B03.tif  B04.tif
            B8A.tif  B11.tif  B12.tif
            SCL.tif
            manifest.json          <- provenance, ignored by the raster engine

Band names are bare on purpose. raster_engine._index_band_directory matches
`(B[0-9A]{2}|SCL)` case-insensitively against the whole filename stem, so a
scene-prefixed name like "S2B_..._B02.tif" can false-match on the platform code.

Bands are kept at NATIVE resolution (10 m for B02/B03/B04, 20 m for the rest).
SceneReader already resamples to a 10 m reference grid per tile; resampling here
would throw away fidelity for no gain.

Usage
-----
    pip install -r requirements-staging.txt

    # Preview what matches, download nothing
    python ingest_pipeline.py --bbox 72.80 18.90 73.05 19.15 \
        --start 2025-01-01 --end 2025-03-31 --dry-run

    # Fetch the two least-cloudy scenes over the AOI
    python ingest_pipeline.py --bbox 72.80 18.90 73.05 19.15 \
        --start 2025-01-01 --end 2025-03-31 --max-cloud 15 --limit 2

Verified against pystac-client 0.9.0 / Earth Search v1 (collection
"sentinel-2-c1-l2a"; "sentinel-2-l2a" is deprecated and no longer updated).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("ingest_pipeline")

# --------------------------------------------------------------------- config

DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-c1-l2a"

# Trinetra band -> Earth Search asset key. Earth Search v1 exposes common names,
# not the ESA "Bxx" identifiers. nir08 is B8A (20 m narrow NIR) - NOT "nir",
# which is the 10 m broad B08 and is the wrong band for Prithvi's HLS stack.
ASSET_KEYS: dict[str, tuple[str, ...]] = {
    "B02": ("blue",),
    "B03": ("green",),
    "B04": ("red",),
    "B8A": ("nir08", "nir09", "nir"),
    "B11": ("swir16",),
    "B12": ("swir22",),
    "SCL": ("scl",),
}

# Bands SceneReader will refuse to open a scene without. Kept here so a partial
# download is detected at fetch time rather than three hours later at ingest.
REQUIRED_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

# GDAL tuning for COG-over-HTTP. READDIR_ON_OPEN=EMPTY_DIR stops GDAL listing
# the whole S3 prefix on every open; the extension allowlist blocks it from
# probing for sidecar files that do not exist. Both cut request count sharply.
# GDAL_HTTP_UNSAFESSL is deliberately NOT set - disabling certificate validation
# on the one machine that touches the public internet is the wrong trade.
#
# GDAL_CACHEMAX must be an int, NOT "256". rasterio special-cases that key and
# routes it to GDALSetCacheMax64(), which takes a C integer, so a str value dies
# with a bare `TypeError: an integer is required` inside rasterio.Env.__enter__
# before a single byte is fetched. Values <= 10000 are read as MiB by rasterio.
# Every other key here is a plain CPLSetConfigOption string.
GDAL_ENV: dict[str, str | int] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_CACHEMAX": 256,
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "33554432",
}

# Refuse absurd AOIs before spending bandwidth. 8000x8000 at 10 m is ~80 km
# square and ~64 MP per band - already well past what an 8 GB card wants to
# tile in one pass. Override with --force if you know what you are doing.
MAX_PIXELS_10M = 8000 * 8000

# AOI bounds are snapped outward onto a lattice this coarse so that the 10 m and
# 20 m crops share a top-left corner. Must be the LCM of every resolution we
# ingest (10 m and 20 m here). Raise to 60.0 if B01/B09/B10 are ever added.
GRID_SNAP_M = 20.0

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 3.0

# Scenes within this many percentage points of the cleanest match count as
# equally usable, and so are all eligible for temporal spreading.
_CLOUD_TOLERANCE_PP = 5.0


class PipelineError(RuntimeError):
    """Fatal, user-actionable pipeline failure."""


# ------------------------------------------------------------------- helpers


def _require_deps() -> tuple[Any, Any, Any, Any, Any]:
    """Import the online-only stack with a readable message if it is absent."""
    try:
        import rasterio
        from pyproj import Transformer
        from pystac_client import Client
        from rasterio.windows import Window, from_bounds
    except ImportError as exc:
        raise PipelineError(
            f"Missing staging dependency: {exc.name}. Install with:\n"
            f"    pip install -r requirements.txt          # rasterio, pyproj\n"
            f"    pip install -r requirements-staging.txt  # pystac-client"
        ) from exc
    return Client, rasterio, from_bounds, Window, Transformer


def _tile_code(item: Any) -> str:
    """
    MGRS tile the item belongs to, e.g. "42RUR".

    Load-bearing for change analysis: two scenes from DIFFERENT MGRS tiles are
    on different grids, and SceneReader.is_grid_compatible will reject the pair.
    An AOI near a tile seam matches items from both, so scenes must be grouped
    by tile before the least-cloudy pick, never picked across tiles.
    """
    props = item.properties or {}
    tile = "".join(
        str(props.get(k, ""))
        for k in ("mgrs:utm_zone", "mgrs:latitude_band", "mgrs:grid_square")
    )
    if tile:
        return tile
    return str(props.get("grid:code", "UNKNOWN")).replace("MGRS-", "")


def _scene_dir_name(item: Any) -> str:
    """
    Build our own directory name rather than trusting item.id.

    raster_engine._parse_safe_timestamp needs `_YYYYmmddTHHMMSS_` bounded by
    underscores to recover the acquisition time. Earth Search item ids do not
    guarantee that shape across collections, so we synthesise it from
    properties, which are stable.
    """
    props = item.properties or {}
    # Earth Search reports platform as "sentinel-2a"; compress it to the ESA
    # short code so scene ids read like real granules in the UI (S2A_T42RUR_...).
    raw = str(props.get("platform", "S2")).lower().replace("-", "")
    platform = f"S2{raw[-1].upper()}" if raw.startswith("sentinel2") else raw.upper()
    dt = item.datetime or datetime.now(timezone.utc)
    return f"{platform}_T{_tile_code(item)}_{dt.strftime('%Y%m%dT%H%M%S')}_L2A"


def _resolve_asset(item: Any, band: str) -> tuple[str, str]:
    """Return (asset_key, href) for a Trinetra band, trying each known alias."""
    for key in ASSET_KEYS[band]:
        asset = item.assets.get(key)
        if asset is not None and asset.href:
            return key, asset.href
    available = ", ".join(sorted(item.assets)) or "<none>"
    raise PipelineError(
        f"Item {item.id} exposes no asset for band {band} "
        f"(tried {ASSET_KEYS[band]}). Available: {available}"
    )


def _vsicurl(href: str) -> str:
    """Wrap an https href for GDAL byte-range streaming."""
    if href.startswith(("/vsicurl/", "/vsis3/")):
        return href
    if href.startswith("s3://"):
        # Earth Search sometimes returns s3:// for the same object; the public
        # HTTPS mirror avoids needing credentials.
        return "/vsicurl/" + href.replace(
            "s3://sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
        )
    return f"/vsicurl/{href}"


def _crop_window(src: Any, bbox_wgs84: tuple[float, float, float, float],
                 from_bounds: Any, Transformer: Any, Window: Any) -> Any:
    """
    Pixel window covering the AOI, reprojected into the asset's own CRS.

    Each band is read in its native CRS/transform, then rounded outward to whole
    pixels and clamped to the dataset. Rounding outward (not nearest) guarantees
    the AOI is fully covered rather than shaved by a fraction of a pixel.
    """
    west, south, east, north = bbox_wgs84
    if src.crs is None:
        raise PipelineError("Remote asset has no CRS; refusing to guess.")

    if src.crs.to_epsg() != 4326:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform([west, east, west, east], [south, south, north, north])
        west, east = min(xs), max(xs)
        south, north = min(ys), max(ys)

    import math

    # Snap the bounds outward onto a GRID_SNAP_M lattice anchored at the GRANULE
    # origin. This is load-bearing, not tidiness.
    #
    # Every asset of one Sentinel-2 granule shares a top-left corner, but the 10 m
    # and 20 m grids step differently. Flooring each asset's window independently
    # against its own grid therefore lands the crops on origins up to 10 m apart -
    # observed exactly that: B02 top at Y=2988010, B8A top at Y=2988020.
    #
    # raster_engine._read_band derives the 20 m window as `row_off / scale`, which
    # silently assumes coincident origins. With a 10 m offset every chip reads its
    # NIR/SWIR triplet half a 20 m pixel north of its RGB triplet - misregistered
    # bands inside Prithvi's six-band stack, with no error raised anywhere.
    #
    # Snapping to 20 m makes the 10 m window's offsets and extent even, so
    # dividing by 2 is exact and both crops share a corner.
    ox, oy = src.transform.c, src.transform.f     # granule left / top
    west = ox + math.floor((west - ox) / GRID_SNAP_M) * GRID_SNAP_M
    east = ox + math.ceil((east - ox) / GRID_SNAP_M) * GRID_SNAP_M
    north = oy - math.floor((oy - north) / GRID_SNAP_M) * GRID_SNAP_M
    south = oy - math.ceil((oy - south) / GRID_SNAP_M) * GRID_SNAP_M

    raw = from_bounds(west, south, east, north, transform=src.transform)

    # Rounded and clamped with plain integer maths rather than Window.round_offsets
    # / .intersection: those helpers changed signature across rasterio 1.3->1.4
    # (the `op=` kwarg and pixel_precision were reworked), and this arithmetic is
    # version-proof. Floor the origin and ceil the far edge so the AOI is fully
    # covered instead of being shaved by a fraction of a pixel.
    col_off = max(0, math.floor(raw.col_off))
    row_off = max(0, math.floor(raw.row_off))
    col_end = min(src.width, math.ceil(raw.col_off + raw.width))
    row_end = min(src.height, math.ceil(raw.row_off + raw.height))

    if col_end <= col_off or row_end <= row_off:
        raise PipelineError(
            "AOI does not overlap this asset's footprint after reprojection."
        )
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)

def _guard_size(width: int, height: int, res_m: float, force: bool) -> None:
    scale = max(res_m / 10.0, 1e-6)
    px_10m = (width * scale) * (height * scale)
    if px_10m > MAX_PIXELS_10M and not force:
        raise PipelineError(
            f"AOI is ~{px_10m / 1e6:.0f} MP at 10 m, over the "
            f"{MAX_PIXELS_10M / 1e6:.0f} MP guard. Shrink --bbox or pass --force."
        )


def _download_band(rasterio: Any, from_bounds: Any, Window: Any, Transformer: Any,
                   href: str, dest: Path,
                   bbox: tuple[float, float, float, float], force: bool) -> dict[str, Any]:
    """Windowed-read one remote COG and write the crop to dest. Returns metadata."""
    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            with rasterio.Env(**GDAL_ENV):
                with rasterio.open(_vsicurl(href)) as src:
                    win = _crop_window(src, bbox, from_bounds, Transformer, Window)
                    _guard_size(int(win.width), int(win.height),
                                abs(src.transform.a), force)

                    # Profile built EXPLICITLY rather than copied from src.profile.
                    # The remote COG's profile carries keys we do not control
                    # (interleave, photometric, sparse/overview hints, nodata
                    # types) and passing them back to the GTiff creation path is
                    # a good way to get an opaque TypeError from GDAL. Only these
                    # keys are needed to write a faithful crop.
                    profile = {
                        "driver": "GTiff",
                        "dtype": src.dtypes[0],
                        "count": 1,
                        "height": int(win.height),
                        "width": int(win.width),
                        "crs": src.crs,
                        "transform": src.window_transform(win),
                        "tiled": True,
                        "blockxsize": 256,
                        "blockysize": 256,
                        "compress": "deflate",
                    }
                    if src.nodata is not None:
                        profile["nodata"] = src.nodata

                    # Single band, streamed straight through - never the full scene.
                    data = src.read(1, window=win)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(dest, "w", **profile) as dst:
                        dst.write(data, 1)

                    return {
                        "href": href,
                        "crs": str(src.crs),
                        "resolution_m": abs(src.transform.a),
                        "width": int(win.width),
                        "height": int(win.height),
                        "origin_x": float(src.window_transform(win).c),
                        "origin_y": float(src.window_transform(win).f),
                        "dtype": str(data.dtype),
                        "bytes": dest.stat().st_size,
                    }
        except PipelineError:
            raise  # geometry / guard failures are not transient
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            # Deterministic programming/API errors. Retrying is pointless and
            # only hides the traceback, so fail immediately and loudly.
            dest.unlink(missing_ok=True)
            logger.debug("non-retryable error on %s", dest.name, exc_info=True)
            raise PipelineError(
                f"{dest.name}: {type(exc).__name__}: {exc} "
                f"(deterministic - not retried; re-run with -v for the traceback)"
            ) from exc
        except Exception as exc:  # rasterio raises RasterioIOError for HTTP faults
            last_exc = exc
            dest.unlink(missing_ok=True)
            logger.debug("transient error on %s", dest.name, exc_info=True)
            if attempt < _RETRY_ATTEMPTS:
                delay = _RETRY_BACKOFF_S * attempt
                logger.warning("  %s attempt %d/%d failed (%s: %s); retrying in %.0fs",
                               dest.name, attempt, _RETRY_ATTEMPTS,
                               type(exc).__name__, exc, delay)
                time.sleep(delay)
    raise PipelineError(f"{dest.name}: all {_RETRY_ATTEMPTS} attempts failed: {last_exc}")


# --------------------------------------------------------------------- search


def search_items(Client: Any, args: argparse.Namespace) -> list[Any]:
    logger.info("Querying %s [%s]", args.stac_url, args.collection)
    try:
        catalog = Client.open(args.stac_url)
        search = catalog.search(
            collections=[args.collection],
            bbox=list(args.bbox),
            datetime=f"{args.start}/{args.end}",
            query={"eo:cloud_cover": {"lt": args.max_cloud}},
            # Metadata-only paging, so this is cheap - but it MUST be large
            # enough to span the whole --start/--end window. Sentinel-2 revisits
            # a tile every ~5 days, so a 3-year range is ~220 acquisitions; a low
            # cap would hand back one end of the range and make --strategy spread
            # silently select across a few months instead of a few years.
            max_items=args.max_items,
        )
        # items() yields lazily. get_items()/get_all_items() are removed in 0.9.
        items = list(search.items())
    except Exception as exc:
        raise PipelineError(
            f"STAC search failed: {exc}. Check network, --stac-url and --collection."
        ) from exc

    if not items:
        raise PipelineError(
            "No scenes matched. Widen --start/--end, raise --max-cloud, "
            "or confirm the AOI is over land covered by Sentinel-2."
        )

    # Group by MGRS tile, then choose ONE tile to draw every scene from. Mixing
    # tiles yields scenes on incompatible grids, which change analysis rejects.
    by_tile: dict[str, list[Any]] = {}
    for it in items:
        by_tile.setdefault(_tile_code(it), []).append(it)

    for scenes in by_tile.values():
        scenes.sort(key=lambda it: (it.properties or {}).get("eo:cloud_cover", 100.0))

    if args.tile:
        tile = args.tile.upper().replace("T", "", 1) if args.tile.upper().startswith("T") \
            else args.tile.upper()
        if tile not in by_tile:
            raise PipelineError(
                f"--tile {args.tile} not among matches. Available: "
                f"{', '.join(sorted(by_tile))}"
            )
    else:
        # Prefer the tile with enough dates to support change analysis; break
        # ties on the cleanest single scene.
        tile = max(
            by_tile,
            key=lambda t: (
                min(len(by_tile[t]), args.limit),
                -(by_tile[t][0].properties or {}).get("eo:cloud_cover", 100.0),
            ),
        )

    if len(by_tile) > 1:
        logger.warning(
            "AOI spans %d MGRS tiles (%s); using T%s only - cross-tile pairs are "
            "not co-registered and change analysis would reject them.",
            len(by_tile), ", ".join(sorted(by_tile)), tile,
        )

    chosen = _select_dates(by_tile[tile], args.limit, args.strategy)
    logger.info("%d scene(s) matched across %d tile(s); T%s has %d, keeping %d (%s)",
                len(items), len(by_tile), tile, len(by_tile[tile]),
                len(chosen), args.strategy)
    return chosen


def _select_dates(scenes: list[Any], limit: int, strategy: str) -> list[Any]:
    """
    Pick `limit` scenes from one tile's candidates.

    "cloud"  - strictly cleanest. Right when you only want one scene.
    "spread" - cleanest-then-widest-apart, and the DEFAULT, because change
               analysis is half this project. Over an arid AOI a batch of
               same-month scenes are all near 0% cloud, so a pure cloud sort
               returns dates days apart and every chip reads as unchanged.
    """
    if limit >= len(scenes):
        return list(scenes)
    if strategy == "cloud" or limit == 1:
        return scenes[:limit]  # already cloud-sorted by the caller

    # Build the clean pool by THRESHOLD, not by top-N. Over an arid AOI a long
    # range yields hundreds of scenes at ~0% cloud; a top-N pool would be an
    # arbitrary handful of them, and spreading across that reaches only a
    # fraction of the requested window. Anything within 5 points of the cleanest
    # scene is equally usable, so all of it is eligible for date spreading.
    best = (scenes[0].properties or {}).get("eo:cloud_cover", 100.0) or 0.0
    ceiling = best + _CLOUD_TOLERANCE_PP
    pool = [s for s in scenes
            if ((s.properties or {}).get("eo:cloud_cover", 100.0) or 0.0) <= ceiling]
    if len(pool) < limit:
        pool = scenes[: max(limit * 4, limit)]
    logger.debug("clean pool: %d/%d scene(s) at <=%.1f%% cloud",
                 len(pool), len(scenes), ceiling)

    pool.sort(key=lambda it: it.datetime or datetime.min.replace(tzinfo=timezone.utc))
    if limit == 2:
        picked = [pool[0], pool[-1]]
    else:
        # Space by ELAPSED TIME, not list index: acquisitions cluster wherever
        # the weather was good, so index spacing would cluster with them.
        t0 = pool[0].datetime
        t1 = pool[-1].datetime
        if t0 and t1 and t1 > t0:
            total = (t1 - t0).total_seconds()
            picked = []
            for i in range(limit):
                target = t0.timestamp() + total * i / (limit - 1)
                cand = min(pool, key=lambda s: abs(s.datetime.timestamp() - target)
                           if s.datetime else float("inf"))
                if cand not in picked:
                    picked.append(cand)
            # Nearest-match can collide on sparse pools; backfill by date.
            for s in pool:
                if len(picked) >= limit:
                    break
                if s not in picked:
                    picked.append(s)
            picked.sort(key=lambda s: s.datetime or datetime.min.replace(tzinfo=timezone.utc))
        else:
            picked = pool[:limit]

    span_days = None
    if picked[0].datetime and picked[-1].datetime:
        span_days = (picked[-1].datetime - picked[0].datetime).days
        if span_days < 60:
            logger.warning(
                "Selected dates span only %d days. Structural change is unlikely "
                "to be visible - widen --start/--end (a 1-2 year gap is ideal for "
                "a change-detection demo).", span_days,
            )
    return picked


def _verify_coregistration(bands_meta: dict[str, Any]) -> None:
    """
    Every band must share one top-left corner, and every coarse band must be an
    exact integer factor of the 10 m reference in both axes.

    raster_engine._read_band derives a coarse band's window by dividing the 10 m
    window offsets by the resolution ratio. That is only valid if the grids share
    an origin. When they do not, the band is read offset by a fraction of a coarse
    pixel and NOTHING raises - the stack is quietly misregistered and the change
    scores drift. Fail the scene here instead.
    """
    ref = bands_meta.get("B02")
    if ref is None:
        return
    for band, m in bands_meta.items():
        if (m["origin_x"], m["origin_y"]) != (ref["origin_x"], ref["origin_y"]):
            raise PipelineError(
                f"{band} origin ({m['origin_x']}, {m['origin_y']}) does not match "
                f"B02 ({ref['origin_x']}, {ref['origin_y']}). Grids are not "
                f"co-registered; raster_engine would read this band offset."
            )
        scale = m["resolution_m"] / ref["resolution_m"]
        if scale != int(scale) or ref["width"] % int(scale) or ref["height"] % int(scale):
            raise PipelineError(
                f"{band} at {m['resolution_m']} m does not divide the 10 m "
                f"reference extent {ref['width']}x{ref['height']} evenly."
            )


def fetch_scene(deps: tuple, item: Any, out_root: Path,
                bbox: tuple[float, float, float, float], force: bool) -> Path | None:
    """Download all bands for one item. Returns the scene dir, or None on failure."""
    Client, rasterio, from_bounds, Window, Transformer = deps
    scene_dir = out_root / _scene_dir_name(item)

    if (scene_dir / "manifest.json").exists():
        logger.info("SKIP %s (already complete)", scene_dir.name)
        return scene_dir

    cloud = (item.properties or {}).get("eo:cloud_cover")
    logger.info("Fetching %s (cloud %.1f%%)", scene_dir.name,
                cloud if cloud is not None else float("nan"))

    bands_meta: dict[str, Any] = {}
    try:
        for band in ASSET_KEYS:
            key, href = _resolve_asset(item, band)
            dest = scene_dir / f"{band}.tif"
            meta = _download_band(rasterio, from_bounds, Window, Transformer,
                                  href, dest, bbox, force)
            meta["asset_key"] = key
            bands_meta[band] = meta
            logger.info("  %-4s %4dx%-4d %5.1f MB  <- %s",
                        band, meta["width"], meta["height"],
                        meta["bytes"] / 1e6, key)

        missing = [b for b in REQUIRED_BANDS if b not in bands_meta]
        if missing:
            raise PipelineError(f"incomplete band set, missing {missing}")

        _verify_coregistration(bands_meta)

        manifest = {
            "scene_id": scene_dir.name,
            "stac_item_id": item.id,
            "collection": item.collection_id,
            "acquired": item.datetime.isoformat() if item.datetime else None,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "aoi_bbox_wgs84": list(bbox),
            "cloud_cover_pct": cloud,
            "platform": (item.properties or {}).get("platform"),
            "bands": bands_meta,
            "tool": "ingest_pipeline.py",
        }
        # Written LAST and atomically: its presence is the completeness marker
        # the watchdog and the resume-skip above both rely on.
        tmp = scene_dir / "manifest.json.part"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(scene_dir / "manifest.json")
        return scene_dir

    except Exception as exc:
        logger.error("FAILED %s: %s", scene_dir.name, exc)
        # A half-downloaded scene would make SceneReader raise on missing bands.
        # Remove it so a re-run is clean.
        shutil.rmtree(scene_dir, ignore_errors=True)
        return None


# ----------------------------------------------------------------------- cli


def _bbox(values: list[float]) -> tuple[float, float, float, float] | None:
    """Validate W S E N. Returns None when invalid so the caller can p.error()."""
    west, south, east, north = values
    if not (-180 <= west < east <= 180) or not (-90 <= south < north <= 90):
        return None
    return west, south, east, north


def _iso_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "secure_drop_zone"
    p = argparse.ArgumentParser(
        description="Fetch Sentinel-2 L2A COG bands into the secure drop zone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="ONLINE tool. Not part of the air-gapped deployment.",
    )
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("W", "S", "E", "N"), help="AOI in WGS84 degrees.")
    p.add_argument("--start", type=_iso_date, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=_iso_date, required=True, help="YYYY-MM-DD")
    p.add_argument("--max-cloud", type=float, default=20.0,
                   help="Server-side eo:cloud_cover ceiling, percent (default 20).")
    p.add_argument("--limit", type=int, default=2,
                   help="Scenes to download, least-cloudy first (default 2).")
    p.add_argument("--max-items", type=int, default=0, metavar="N",
                   help="Cap on STAC items inspected. Default 0 = auto-size from "
                        "the date range (~1 item per 2 days, min 100, max 2000). "
                        "Metadata only, so a large value costs little.")
    p.add_argument("--strategy", choices=("spread", "cloud"), default="spread",
                   help="spread (default): cleanest scenes, then widest apart in "
                        "time - what change analysis needs. cloud: strictly "
                        "least-cloudy, ignoring temporal gap.")
    p.add_argument("--tile", default=None, metavar="MGRS",
                   help="Force one MGRS tile, e.g. 42RUR. Default picks the tile "
                        "with the most usable dates. All scenes come from a "
                        "single tile so change analysis stays co-registered.")
    p.add_argument("--out", type=Path, default=default_out,
                   help=f"Drop zone (default {default_out}).")
    p.add_argument("--stac-url", default=DEFAULT_STAC_URL)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--dry-run", action="store_true",
                   help="List matches and exit without downloading.")
    p.add_argument("--force", action="store_true",
                   help="Bypass the AOI size guard.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    bbox = _bbox(args.bbox)
    if bbox is None:
        p.error(f"--bbox must be W S E N with W<E and S<N; got {args.bbox}")
    args.bbox = bbox
    if args.start > args.end:
        p.error("--start must not be after --end")
    if args.limit < 1:
        p.error("--limit must be >= 1")
    if args.max_items < 0:
        p.error("--max-items must be >= 0")
    if args.max_items == 0:
        span = (datetime.strptime(args.end, "%Y-%m-%d")
                - datetime.strptime(args.start, "%Y-%m-%d")).days
        args.max_items = max(100, min(2000, span // 2))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    # basicConfig sets the ROOT level, so -v otherwise turns on rasterio's and
    # urllib3's own DEBUG streams and buries our traceback in GDAL/curl chatter.
    for noisy in ("rasterio", "rasterio._env", "rasterio._io", "urllib3",
                  "botocore", "pystac", "pystac_client", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        deps = _require_deps()
        Client = deps[0]
        items = search_items(Client, args)

        if args.dry_run:
            for it in items:
                props = it.properties or {}
                logger.info("%-40s T%-6s cloud %5.1f%%  %s",
                            _scene_dir_name(it),
                            _tile_code(it),
                            props.get("eo:cloud_cover", float("nan")),
                            it.datetime.date() if it.datetime else "?")
            logger.info("Dry run - nothing downloaded.")
            return 0

        args.out.mkdir(parents=True, exist_ok=True)
        ok = [d for it in items
              if (d := fetch_scene(deps, it, args.out, args.bbox, args.force))]

        logger.info("Done: %d/%d scene(s) in %s", len(ok), len(items), args.out)
        for d in ok:
            logger.info("  %s", d.name)
        if len(ok) < 2:
            logger.warning(
                "Change analysis needs two co-registered dates over the same "
                "AOI. Re-run with a wider date range to get a second scene."
            )
        return 0 if ok else 1

    except PipelineError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
