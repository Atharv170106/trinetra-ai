"""
Generate two co-registered synthetic Sentinel-2-like scenes for offline demos.

Why this exists: every test so far has used throwaway scenes inside a tmpdir, so
`backend/sample_data/` is empty and the dashboard has nothing to retrieve. Real
Copernicus L2A products are ~1 GB and cannot be fetched on an air-gapped host,
so until a real granule is sideloaded this stands in for one.

What it produces (default 1024x1024 -> a 4x4 grid of 256 px chips):

    sample_data/DEMO_T43RGN_20240115T052131_L2A/   <- T1
    sample_data/DEMO_T43RGN_20260115T052131_L2A/   <- T2
        B02.tif B03.tif B04.tif        10 m, 1024x1024 uint16
        B05.tif B06.tif B07.tif SCL.tif  20 m,  512x512 uint16

The scenes are NOT noise. Four land-cover classes are painted with plausible
L2A reflectance DNs so that text retrieval is actually discriminative - a query
for water should not rank a built-up chip first. Both scenes share one CRS,
size and geotransform, so the co-registration guard in compare_scenes() passes.

Planted ground truth in T2 (a 2-year gap):
    r2c1, r2c2  a new bright concrete pad / airstrip on former scrubland
    r3c3        the reservoir has shrunk by ~55%
    r0c3        opaque cloud (SCL 9) - must be REJECTED at ingest and
                SKIPPED by change analysis, not reported as change

Everything else differs only by low-amplitude sensor/atmospheric noise, which
is what sets the "unchanged" baseline score.

Usage (from the repo root, with the venv active):
    .\\venv\\Scripts\\python.exe backend\\make_demo_scenes.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:  # pragma: no cover
    sys.exit("rasterio is not installed. Activate the venv first.")

logger = logging.getLogger("make_demo_scenes")

CRS = "EPSG:32643"          # UTM 43N - covers much of northern India
ORIGIN_X, ORIGIN_Y = 300_000.0, 3_200_000.0
REF_RES_M = 10
BANDS_10M = ("B02", "B03", "B04")
BANDS_20M = ("B05", "B06", "B07")

# Land-cover classes -> per-band reflectance DN (L2A scale factor 10000).
# Ordered B02,B03,B04,B05,B06,B07 (blue, green, red, red-edge x2, NIR-ish).
SOIL, VEG, WATER, URBAN, CONCRETE, CLOUD = 0, 1, 2, 3, 4, 5
SPECTRA: dict[int, tuple[int, ...]] = {
    SOIL:     (1300, 1600, 2100, 2500, 2900, 3000),
    VEG:      (380, 700, 420, 1400, 2800, 3200),
    WATER:    (900, 700, 450, 350, 250, 200),
    URBAN:    (2100, 2300, 2500, 2700, 2900, 3000),
    CONCRETE: (3400, 3550, 3700, 3800, 3900, 3950),
    CLOUD:    (7600, 7700, 7800, 7900, 8000, 8100),
}
# Sentinel-2 Scene Classification Layer codes the ingest pipeline reacts to.
SCL_FOR = {
    SOIL: 5,        # NOT_VEGETATED
    VEG: 4,         # VEGETATION
    WATER: 6,       # WATER
    URBAN: 5,
    CONCRETE: 5,
    CLOUD: 9,       # CLOUD_HIGH_PROBABILITY -> rejected by the SCL mask
}


# ------------------------------------------------------------------- textures
def _box3(a: np.ndarray) -> np.ndarray:
    """One 3x3 mean pass. Edge-padded so the frame does not darken."""
    p = np.pad(a, 1, mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def _smooth_field(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """A 0..1 low-frequency field: coarse random grid, upsampled and blurred."""
    coarse = rng.random((cells, cells), dtype=np.float32)
    block = (int(np.ceil(h / cells)), int(np.ceil(w / cells)))
    field = np.kron(coarse, np.ones(block, dtype=np.float32))[:h, :w]
    for _ in range(6):
        field = _box3(field)
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / (hi - lo) if hi > lo else field


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


# --------------------------------------------------------------- label raster
def build_labels(size: int, *, epoch: int, rng: np.random.Generator) -> np.ndarray:
    """
    Paint the land-cover map. `epoch` 1 or 2 selects the ground truth above.

    Scale factors are relative to the default 1024 so a --size override still
    lands the planted features inside the intended chips.
    """
    s = size / 1024.0
    labels = np.full((size, size), SOIL, dtype=np.uint8)

    # Vegetation: a natural-looking blob, thresholded low-frequency noise.
    veg = _smooth_field(size, size, 16, rng) > 0.56
    labels[veg] = VEG

    # Built-up district with a street grid - high-contrast texture, which is
    # what makes "urban" separable from "bare ground" for RemoteCLIP.
    r0, r1 = int(40 * s), int(430 * s)
    c0, c1 = int(60 * s), int(470 * s)
    labels[r0:r1, c0:c1] = URBAN
    street = max(6, int(10 * s))
    pitch = max(24, int(64 * s))
    for r in range(r0, r1, pitch):
        labels[r : r + street, c0:c1] = SOIL
    for c in range(c0, c1, pitch):
        labels[r0:r1, c : c + street] = SOIL

    # A meandering river, drawn last so it cuts through everything.
    rows = np.arange(size)
    centre = size / 2 + (260 * s) * np.sin(2 * np.pi * rows / (900 * s))
    half = max(8, int(17 * s))
    cols = np.arange(size)[None, :]
    labels[np.abs(cols - centre[:, None]) < half] = WATER

    # Reservoir - shrinks in T2. This is planted change, not noise.
    # Kept clear of the 768 px chip boundary so the change lands in r3c3 alone;
    # bleeding into r3c2 would report a chip the ground truth does not claim.
    shrink = 1.0 if epoch == 1 else 0.45
    labels[_ellipse(size, size, 860 * s, 890 * s, 78 * s * shrink, 92 * s * shrink)] = WATER

    if epoch == 2:
        # New concrete pad spanning two chips in grid row 2.
        labels[int(540 * s) : int(700 * s), int(300 * s) : int(620 * s)] = CONCRETE
        # Opaque cloud, contained within chip r0c3.
        labels[_ellipse(size, size, 118 * s, 916 * s, 128 * s, 128 * s)] = CLOUD

    return labels


# ------------------------------------------------------------------- painting
def render_band(
    labels: np.ndarray, band_index: int, rng: np.random.Generator, shade: np.ndarray
) -> np.ndarray:
    """
    One band as uint16 DN.

    Three noise terms, each deliberate: `shade` is shared terrain illumination
    (identical across bands, so the bands stay correlated like real optics);
    a small independent speckle keeps per-chip stddev above min_stddev so no
    chip is discarded as flat; and the caller reseeds per epoch, which is what
    produces a realistic non-zero "unchanged" change score.
    """
    out = np.zeros(labels.shape, dtype=np.float32)
    for cls, spectrum in SPECTRA.items():
        mask = labels == cls
        if mask.any():
            out[mask] = spectrum[band_index]
    out *= 0.90 + 0.20 * shade
    out += rng.normal(0.0, 55.0, size=labels.shape).astype(np.float32)
    return np.clip(out, 1.0, 10_000.0).astype(np.uint16)


def downsample(arr: np.ndarray) -> np.ndarray:
    """10 m -> 20 m by 2x2 mean. Matches how the engine rescales on read."""
    h, w = arr.shape
    view = arr[: h // 2 * 2, : w // 2 * 2].astype(np.float32)
    return view.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.uint16)


def downsample_mode(arr: np.ndarray) -> np.ndarray:
    """SCL must never be averaged - class 4 and 6 do not average to anything."""
    return arr[::2, ::2].copy()


def write_raster(path: Path, data: np.ndarray, res_m: int) -> None:
    """Tiled + compressed, so windowed reads stay cheap on a laptop."""
    h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint16",
        crs=CRS,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, res_m, res_m),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(data, 1)


def build_scene(out_dir: Path, *, size: int, epoch: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Terrain seed is shared so both dates describe the same ground; the
    # radiometric seed differs so the two dates are not bit-identical.
    terrain_rng = np.random.default_rng(seed)
    labels = build_labels(size, epoch=epoch, rng=terrain_rng)
    shade = _smooth_field(size, size, 32, np.random.default_rng(seed + 1))
    radio_rng = np.random.default_rng(seed + 1000 * epoch)

    for i, band in enumerate(BANDS_10M):
        write_raster(out_dir / f"{band}.tif", render_band(labels, i, radio_rng, shade), REF_RES_M)

    for offset, band in enumerate(BANDS_20M):
        full = render_band(labels, 3 + offset, radio_rng, shade)
        write_raster(out_dir / f"{band}.tif", downsample(full), REF_RES_M * 2)
        del full

    scl = np.zeros(labels.shape, dtype=np.uint16)
    for cls, code in SCL_FOR.items():
        scl[labels == cls] = code
    write_raster(out_dir / "SCL.tif", downsample_mode(scl), REF_RES_M * 2)

    grid = size // 256
    logger.info("wrote %s  (%dx%d, %dx%d chip grid)", out_dir.name, size, size, grid, grid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_data",
        help="destination directory (default: backend/sample_data)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="scene edge in 10 m pixels, multiple of 256 (default 1024). The "
        "planted features scale with the scene while chips stay 256 px, so the "
        "documented chip IDs only hold at 1024.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.size % 256:
        return int(bool(logger.error("--size must be a multiple of 256")) or 2)
    if args.size != 1024:
        logger.warning(
            "--size %d: features stay proportional but will not align to the "
            "256 px chip grid, so the expected chip IDs below will differ.",
            args.size,
        )

    scenes = {1: "DEMO_T43RGN_20240115T052131_L2A", 2: "DEMO_T43RGN_20260115T052131_L2A"}
    for epoch, name in scenes.items():
        build_scene(args.out / name, size=args.size, epoch=epoch, seed=args.seed)

    logger.info(
        "\nIngest both from the dashboard's Ingest tab (source = the folder name),\n"
        "then compare T1 -> T2 on the Change tab. Expected: a new concrete pad at\n"
        "r2c1/r2c2, a shrunken reservoir at r3c3, and r0c3 skipped as cloudy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
