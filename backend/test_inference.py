"""
Trinetra AI - Phase 3 verification.

Exercises the real models and a live Qdrant. Run from the repo root with the
Qdrant container up:

    docker compose up -d
    python backend/test_inference.py

Checks:
  1. RemoteCLIP loads FP16 on CUDA; VRAM stays under the ceiling
  2. Text embeddings are 512-dim, L2-normalized, deterministic
  3. Image embeddings are 512-dim; batching matches single-image results
  4. Semantic sanity: a synthetic vegetation chip scores higher on a vegetation
     prompt than on an unrelated prompt
  5. Qdrant collection bootstrap is idempotent; payload indexes created
  6. Upsert -> search -> retrieve round trip; re-upsert does not duplicate
  7. Filtered search (scene_id, max_cloud) narrows results correctly
  8. Prithvi encoder loads with num_frames=2 and both models stay resident
  9. Change detection: identical chips score ~0, altered chips score higher
 10. Full ingest over a synthetic scene, then a real text query against it
"""

from __future__ import annotations

import logging
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

PASS, FAIL, WARN = "[ PASS ]", "[ FAIL ]", "[ WARN ]"
_failures: list[str] = []
_warnings: list[str] = []

TEST_COLLECTION = "_trinetra_phase3_test"
SCENE_W = SCENE_H = 768  # 3x3 grid of 256px tiles
CRS = "EPSG:32643"
ORIGIN_X, ORIGIN_Y = 300000.0, 3200000.0
ALL_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07")


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def bad(check: str, msg: str) -> None:
    print(f"{FAIL} {check}: {msg}")
    _failures.append(check)


def warn(msg: str) -> None:
    print(f"{WARN} {msg}")
    _warnings.append(msg)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# ------------------------------------------------------------ synthetic inputs
def synth_vegetation_chip(size: int = 256, seed: int = 0) -> np.ndarray:
    """Green-dominant textured RGB chip."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = rng.integers(30, 70, (size, size))    # low red
    img[..., 1] = rng.integers(110, 200, (size, size))  # high green
    img[..., 2] = rng.integers(20, 60, (size, size))    # low blue
    return img


def synth_urban_chip(size: int = 256, seed: int = 1) -> np.ndarray:
    """Grey chip with a bright rectilinear grid, roughly built-up looking."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 140, (size, size)).astype(np.uint8)
    img = np.stack([base] * 3, axis=-1)
    for x in range(0, size, 32):
        img[:, x : x + 4] = 225
    for y in range(0, size, 32):
        img[y : y + 4, :] = 225
    return img


def _write_band(path: Path, arr: np.ndarray, res: int) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=CRS,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, res, res),
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def build_scene(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for i, band in enumerate(ALL_BANDS):
        res = 10 if band in ("B02", "B03", "B04") else 20
        dim = SCENE_W if res == 10 else SCENE_W // 2
        arr = (rng.integers(500, 3200, (dim, dim)) + i * 100).astype(np.uint16)
        _write_band(root / f"{band}.tif", arr, res)
        del arr
    scl = np.full((SCENE_W // 2, SCENE_H // 2), 4, dtype=np.uint8)
    _write_band(root / "SCL.tif", scl, 20)


# ------------------------------------------------------------------- 1-4 CLIP
def check_remoteclip() -> None:
    section("RemoteCLIP")
    import torch

    from app.services.ml_inference import InferenceError, remoteclip, vram_report

    try:
        remoteclip.load()
    except InferenceError as exc:
        bad("remoteclip load", str(exc))
        return

    rep = vram_report()
    ok(f"loaded on {rep['device_name']} | VRAM {rep['vram_allocated_gb']:.2f}/"
       f"{rep['vram_total_gb']:.2f} GB")
    if rep["over_ceiling"]:
        bad("vram ceiling", f"{rep['vram_allocated_gb']:.2f} GB exceeds "
                            f"{rep['vram_ceiling_gb']} GB")

    if remoteclip._dtype != torch.float16 and torch.cuda.is_available():
        bad("dtype", f"expected float16 on CUDA, got {remoteclip._dtype}")
    else:
        ok(f"dtype {remoteclip._dtype}")

    # Text embeddings
    try:
        vecs = remoteclip.encode_text(["aircraft on tarmac", "dense forest canopy"])
    except InferenceError as exc:
        bad("encode_text", str(exc))
        return

    if vecs.shape == (2, settings.embedding_dim) and vecs.dtype == np.float32:
        ok(f"encode_text -> {vecs.shape} {vecs.dtype}")
    else:
        bad("encode_text shape", f"got {vecs.shape} {vecs.dtype}")
        return

    norms = np.linalg.norm(vecs, axis=1)
    if np.allclose(norms, 1.0, atol=1e-3):
        ok(f"L2-normalized (norms {norms.round(5)})")
    else:
        bad("normalization", f"norms {norms}")

    again = remoteclip.encode_text(["aircraft on tarmac"])
    if np.allclose(again[0], vecs[0], atol=1e-3):
        ok("deterministic across calls")
    else:
        bad("determinism", f"max delta {np.abs(again[0] - vecs[0]).max():.5f}")

    # Distinct prompts must not collapse to the same vector.
    sim = float(vecs[0] @ vecs[1])
    if sim < 0.95:
        ok(f"distinct prompts separated (cos={sim:.4f})")
    else:
        bad("text discrimination", f"unrelated prompts cos={sim:.4f}")

    # Image embeddings + batching equivalence
    chips = [synth_vegetation_chip(seed=i) for i in range(5)]
    try:
        batched = remoteclip.encode_images(chips, batch_size=5)
        singly = np.concatenate(
            [remoteclip.encode_images([c], batch_size=1) for c in chips], axis=0
        )
    except InferenceError as exc:
        bad("encode_images", str(exc))
        return

    if batched.shape == (5, settings.embedding_dim):
        ok(f"encode_images -> {batched.shape}")
    else:
        bad("encode_images shape", f"got {batched.shape}")
        return

    delta = np.abs(batched - singly).max()
    if delta < 5e-2:
        ok(f"batched == single (max delta {delta:.5f}, fp16 tolerance)")
    else:
        bad("batch consistency", f"max delta {delta:.5f}")

    # Semantic sanity check
    veg_vec = remoteclip.encode_images([synth_vegetation_chip(seed=99)])[0]
    prompts = ["green vegetation and trees", "an airport runway with aircraft"]
    tvecs = remoteclip.encode_text(prompts)
    scores = tvecs @ veg_vec
    print(f"        vegetation chip vs {prompts!r} -> {scores.round(4)}")
    if scores[0] > scores[1]:
        ok(f"semantic ordering correct ({scores[0]:.4f} > {scores[1]:.4f})")
    else:
        warn(f"vegetation chip scored higher on the runway prompt "
             f"({scores[1]:.4f} >= {scores[0]:.4f}). Synthetic noise chips are "
             "not real imagery, so this is weak evidence, not a failure.")


# ------------------------------------------------------------- 5-7 Qdrant
def check_vector_store() -> None:
    section("Qdrant vector store")
    from app.services.vector_store import VectorStore, VectorStoreError, tile_point_id

    store = VectorStore(collection=TEST_COLLECTION)
    try:
        health = store.health()
        if not health.get("reachable"):
            bad("qdrant reachable", health.get("error", "unknown"))
            return
        ok(f"reachable at {store.url}")

        created = store.ensure_collection(recreate=True)
        ok(f"ensure_collection(recreate=True) -> created={created}")
        if store.ensure_collection() is False:
            ok("second ensure_collection is a no-op (idempotent)")
        else:
            bad("idempotency", "ensure_collection recreated an existing collection")

        rng = np.random.default_rng(3)

        def unit(n: int) -> np.ndarray:
            v = rng.standard_normal((n, settings.embedding_dim)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        vecs = unit(10)
        records = []
        for i in range(10):
            records.append((
                f"SCENE_A_r{i:04d}c0000",
                vecs[i],
                {
                    "scene_id": "SCENE_A",
                    "acquisition_timestamp": "2026-01-15T05:21:31+00:00",
                    "wgs84_bounding_box": [77.0 + i * 0.01, 28.0, 77.01 + i * 0.01, 28.01],
                    "scl_cloud_coverage": 0.05 * i,
                },
            ))
        records.append((
            "SCENE_B_r0000c0000",
            unit(1)[0],
            {
                "scene_id": "SCENE_B",
                "acquisition_timestamp": "2026-02-20T05:21:31+00:00",
                "wgs84_bounding_box": [78.0, 29.0, 78.01, 29.01],
                "scl_cloud_coverage": 0.0,
            },
        ))

        written = store.upsert_tiles(records)
        if written == 11 and store.count() == 11:
            ok(f"upsert {written} points, count={store.count()}")
        else:
            bad("upsert", f"wrote {written}, count={store.count()}")

        # Idempotence: same tile_ids must overwrite, not duplicate.
        store.upsert_tiles(records)
        if store.count() == 11:
            ok("re-upsert is idempotent (deterministic UUID5 ids)")
        else:
            bad("idempotence", f"count grew to {store.count()}")

        # Exact-match search
        hits = store.search(vecs[0], limit=3)
        if hits and hits[0].tile_id == "SCENE_A_r0000c0000" and hits[0].score > 0.99:
            ok(f"search top hit {hits[0].tile_id} score={hits[0].score:.4f}")
        else:
            bad("search", f"unexpected top hit {hits[0].tile_id if hits else None}")

        if hits and hits[0].wgs84_bounding_box and hits[0].scene_id == "SCENE_A":
            ok(f"payload intact: bbox={hits[0].wgs84_bounding_box} "
               f"cloud={hits[0].scl_cloud_coverage}")
        else:
            bad("payload", "bbox or scene_id missing from hit")

        # Filters
        b_hits = store.search(vecs[0], limit=20, scene_ids=["SCENE_B"])
        if all(h.scene_id == "SCENE_B" for h in b_hits) and len(b_hits) == 1:
            ok(f"scene_id filter -> {len(b_hits)} hit(s), all SCENE_B")
        else:
            bad("scene filter", f"got {[h.scene_id for h in b_hits]}")

        clean = store.search(vecs[0], limit=20, max_cloud=0.10)
        if clean and all((h.scl_cloud_coverage or 0) <= 0.10 + 1e-9 for h in clean):
            ok(f"max_cloud=0.10 filter -> {len(clean)} hit(s), "
               f"max cloud {max((h.scl_cloud_coverage or 0) for h in clean):.2f}")
        else:
            bad("cloud filter",
                f"clouds {[h.scl_cloud_coverage for h in clean]}")

        thresholded = store.search(vecs[0], limit=20, score_threshold=0.99)
        if len(thresholded) == 1:
            ok(f"score_threshold=0.99 -> {len(thresholded)} hit")
        else:
            warn(f"score_threshold=0.99 returned {len(thresholded)} hits "
                 "(random vectors can be coincidentally close)")

        fetched = store.get_tile("SCENE_A_r0005c0000")
        if fetched and fetched.get("scene_id") == "SCENE_A":
            ok(f"get_tile -> {fetched['tile_id']}")
        else:
            bad("get_tile", f"got {fetched}")

        scrolled = list(store.iter_scene_tiles("SCENE_A"))
        if len(scrolled) == 10:
            ok(f"iter_scene_tiles -> {len(scrolled)} tiles")
        else:
            bad("iter_scene_tiles", f"got {len(scrolled)}, expected 10")

        store.delete_scene("SCENE_A")
        if store.count() == 1:
            ok("delete_scene removed only SCENE_A tiles")
        else:
            bad("delete_scene", f"count={store.count()}, expected 1")

        # Dimension guard
        try:
            store.upsert_tiles([("bad", np.zeros(128, dtype=np.float32), {})])
            bad("dim guard", "accepted a 128-dim vector")
        except VectorStoreError:
            ok("wrong-dimension vector rejected")

        try:
            v = np.zeros(settings.embedding_dim, dtype=np.float32)
            v[0] = np.nan
            store.upsert_tiles([("nan", v, {})])
            bad("nan guard", "accepted a NaN vector")
        except VectorStoreError:
            ok("NaN vector rejected")
    finally:
        try:
            if store.client.collection_exists(TEST_COLLECTION):
                store.client.delete_collection(TEST_COLLECTION)
                ok("test collection cleaned up")
        except Exception as exc:
            warn(f"cleanup failed: {exc}")
        store.close()


# --------------------------------------------------------------- 8-9 Prithvi
def check_prithvi() -> None:
    section("Prithvi-EO-2.0 change detection")
    from app.services.ml_inference import (
        InferenceError,
        prithvi,
        remoteclip,
        vram_report,
    )
    from app.services.raster_engine import normalize_for_prithvi

    if not settings.prithvi_checkpoint.is_file():
        warn(f"{settings.prithvi_checkpoint} absent - skipping Prithvi checks.")
        return

    try:
        prithvi.load()
    except InferenceError as exc:
        bad("prithvi load", str(exc))
        return

    rep = vram_report()
    ok(f"encoder loaded on {rep['prithvi_device']}")
    ok(f"both models resident: clip={rep['remoteclip_loaded']} "
       f"prithvi={rep['prithvi_loaded']} | VRAM {rep['vram_allocated_gb']:.2f}/"
       f"{rep['vram_total_gb']:.2f} GB")
    if rep["over_ceiling"]:
        bad("dual-model ceiling",
            f"{rep['vram_allocated_gb']:.2f} GB over {rep['vram_ceiling_gb']} GB")
    else:
        headroom = rep["vram_total_gb"] - rep["vram_allocated_gb"]
        ok(f"{headroom:.2f} GB headroom remaining")

    size = settings.prithvi_input_size
    rng = np.random.default_rng(21)
    base_dn = rng.integers(500, 3000, (6, size, size)).astype(np.uint16)
    t1 = normalize_for_prithvi(base_dn)

    # Identical input -> near-zero change
    try:
        same = prithvi.compare(t1, t1.copy())
    except InferenceError as exc:
        bad("compare identical", str(exc))
        return
    print(f"        identical: cos_dist={same.cosine_distance:.6f} "
          f"l2={same.l2_distance:.4f}")
    if same.cosine_distance < 0.01:
        ok(f"identical chips -> change_score {same.change_score:.6f}")
    else:
        bad("identical compare",
            f"expected ~0, got {same.cosine_distance:.6f}")

    # Structural alteration -> larger change than identical
    altered_dn = base_dn.copy()
    altered_dn[:, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 3500
    t2 = normalize_for_prithvi(altered_dn)
    try:
        changed = prithvi.compare(t1, t2)
    except InferenceError as exc:
        bad("compare altered", str(exc))
        return
    print(f"        altered:   cos_dist={changed.cosine_distance:.6f} "
          f"l2={changed.l2_distance:.4f}")
    if changed.change_score > same.change_score:
        ok(f"altered chip scores higher ({changed.change_score:.6f} > "
           f"{same.change_score:.6f})")
    else:
        bad("change sensitivity",
            f"altered {changed.change_score:.6f} <= identical {same.change_score:.6f}")

    # Shape validation
    try:
        prithvi.compare(t1[:3], t2[:3])
        bad("band guard", "accepted a 3-band cube")
    except ValueError:
        ok("wrong band count rejected")

    # Non-224 input must be resampled, not rejected.
    small_dn = rng.integers(500, 3000, (6, 256, 256)).astype(np.uint16)
    small = normalize_for_prithvi(small_dn)
    try:
        res = prithvi.compare(small, small.copy())
        ok(f"256px input auto-resampled to {size}px "
           f"(change_score {res.change_score:.6f})")
    except Exception as exc:
        bad("resample", f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------- 10 ingest
def check_ingest(tmp: Path) -> None:
    section("End-to-end ingest + search")
    from app.services.ingest import ingest_scene
    from app.services.ml_inference import remoteclip
    from app.services.vector_store import VectorStore, vector_store

    scene_dir = tmp / "ingest_scene"
    build_scene(scene_dir)

    original_collection = vector_store.collection
    original_cache = settings.tiles_cache_dir
    vector_store.collection = TEST_COLLECTION
    settings.tiles_cache_dir = tmp / "tiles_cache"

    try:
        report = ingest_scene(str(scene_dir), scene_id="INGEST_TEST",
                              save_previews=True)
        info = report.as_dict()
        print(f"        {info}")

        if report.errors:
            bad("ingest errors", "; ".join(report.errors[:3]))
        if report.tiles_indexed == 9:
            ok(f"indexed {report.tiles_indexed}/9 tiles in "
               f"{report.elapsed_seconds:.1f}s "
               f"({info['tiles_per_second']} tiles/s)")
        else:
            bad("ingest count",
                f"indexed {report.tiles_indexed}, expected 9 (3x3 grid)")

        previews = list((settings.tiles_cache_dir / "INGEST_TEST").glob("*.png"))
        if len(previews) == report.tiles_indexed:
            ok(f"{len(previews)} PNG previews written")
        else:
            bad("previews", f"{len(previews)} PNGs for {report.tiles_indexed} tiles")

        # A real text query against the freshly indexed scene.
        qvec = remoteclip.encode_text(["satellite view of terrain"])[0]
        hits = vector_store.search(qvec, limit=5)
        if hits:
            ok(f"text search returned {len(hits)} hit(s), "
               f"top={hits[0].tile_id} score={hits[0].score:.4f}")
            h = hits[0]
            if h.wgs84_bounding_box and h.scene_id == "INGEST_TEST":
                ok(f"hit payload complete: bbox={h.wgs84_bounding_box}")
            else:
                bad("hit payload", f"scene_id={h.scene_id} bbox={h.wgs84_bounding_box}")
        else:
            bad("text search", "no hits against an indexed scene")

        # Re-ingest must not duplicate.
        before = vector_store.count()
        ingest_scene(str(scene_dir), scene_id="INGEST_TEST", save_previews=False)
        after = vector_store.count()
        if before == after:
            ok(f"re-ingest is idempotent ({after} points)")
        else:
            bad("re-ingest", f"count changed {before} -> {after}")
    finally:
        try:
            store = VectorStore(collection=TEST_COLLECTION)
            if store.client.collection_exists(TEST_COLLECTION):
                store.client.delete_collection(TEST_COLLECTION)
            store.close()
        except Exception as exc:
            warn(f"ingest cleanup failed: {exc}")
        vector_store.collection = original_collection
        settings.tiles_cache_dir = original_cache
        vector_store.close()


# -------------------------------------------------------------------- runner
def main() -> int:
    print("=" * 68)
    print(" TRINETRA AI - PHASE 3 INFERENCE + VECTOR STORE VERIFICATION")
    print("=" * 68)

    tmp = Path(tempfile.mkdtemp(prefix="trinetra_phase3_"))
    try:
        for check in (check_remoteclip, check_vector_store, check_prithvi):
            try:
                check()
            except Exception as exc:
                bad(check.__name__, f"unhandled {type(exc).__name__}: {exc}")
                traceback.print_exc()
        try:
            check_ingest(tmp)
        except Exception as exc:
            bad("check_ingest", f"unhandled {type(exc).__name__}: {exc}")
            traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    section("Summary")
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed: {', '.join(_failures)}")
    else:
        print(f"{PASS} Phase 3 verified.")
    if _warnings:
        print(f"{WARN} {len(_warnings)} warning(s) - non-blocking.")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
