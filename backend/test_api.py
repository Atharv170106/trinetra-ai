"""
Trinetra AI - Phase 4 verification.

Drives the FastAPI app in-process via TestClient (no uvicorn needed), against a
throwaway Qdrant collection and two synthetic co-registered scenes.

    docker compose up -d
    .\\venv\\Scripts\\python.exe backend\\test_api.py

Checks:
  1.  / and /api/health report status, weights, GPU
  2.  Schema validation rejects bad input (422) before any GPU work
  3.  POST /api/ingest indexes a synthetic scene; 404 on a bad path
  4.  POST /api/search returns ranked hits with bbox + preview_url
  5.  Search filters (scene_ids, max_cloud, limit cap) behave
  6.  GET /api/scenes aggregates the indexed scene
  7.  GET /api/tiles/{id} serves PNG; path traversal is refused
  8.  POST /api/temporal-change scores a real T1/T2 pair
  9.  Non-co-registered scenes are rejected with 422
 10.  /temporal-change/export emits valid GeoJSON
 11.  /api/triage round trip: record, read back, verdict supersedes
 12.  /api/export produces annotated GeoJSON; unknown tiles reported
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

PASS, FAIL, WARN = "[ PASS ]", "[ FAIL ]", "[ WARN ]"
_failures: list[str] = []
_warnings: list[str] = []

TEST_COLLECTION = "_trinetra_phase4_test"
DIM = 768  # 3x3 grid of 256px tiles
CRS = "EPSG:32643"
ORIGIN_X, ORIGIN_Y = 300000.0, 3200000.0
BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")


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


# ------------------------------------------------------------ synthetic scenes
def _write(path: Path, arr: np.ndarray, res: int, origin=(ORIGIN_X, ORIGIN_Y)) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=CRS,
        transform=from_origin(origin[0], origin[1], res, res),
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def build_scene(root: Path, *, seed: int, alter_cell: tuple[int, int] | None = None,
                flat_cell: tuple[int, int] | None = None,
                origin=(ORIGIN_X, ORIGIN_Y)) -> None:
    """
    3x3-tile L2A-like scene.

    alter_cell paints a bright but textured block into one grid cell on every
    band, so change detection has a real signal to find.
    flat_cell paints a constant block - stddev 0 - to prove the flat-chip filter
    does not suppress change detection.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, band in enumerate(BANDS):
        res = 10 if band in ("B02", "B03", "B04") else 20
        dim = DIM if res == 10 else DIM // 2
        arr = (rng.integers(600, 3000, (dim, dim)) + i * 80).astype(np.uint16)
        if alter_cell is not None:
            r, c = alter_cell
            step = 256 if res == 10 else 128
            # Bright but still textured. A constant block would read as stddev 0
            # and get filtered as a flat chip, which is not what we are testing.
            patch = rng.integers(3600, 4000, (step, step)).astype(np.uint16)
            arr[r * step : (r + 1) * step, c * step : (c + 1) * step] = patch
            del patch
        if flat_cell is not None:
            r, c = flat_cell
            step = 256 if res == 10 else 128
            arr[r * step : (r + 1) * step, c * step : (c + 1) * step] = 3000
        _write(root / f"{band}.tif", arr, res, origin)
        del arr
    scl = np.full((DIM // 2, DIM // 2), 4, dtype=np.uint8)
    _write(root / "SCL.tif", scl, 20, origin)


# --------------------------------------------------------------- 1-2 basics
def check_basics(client) -> None:
    section("App bootstrap + validation")

    # The endpoint catalogue lives at /api, not /. "/" is either the built SPA's
    # index.html (not JSON at all) or the dev-mode JSON stub, so asserting
    # "endpoints" in GET / could never pass in either configuration.
    r = client.get("/api")
    if r.status_code == 200 and "endpoints" in r.json():
        ok(f"GET /api -> {len(r.json()['endpoints'])} endpoints advertised")
    else:
        bad("api root", f"{r.status_code} {r.text[:120]}")

    # "/" is mode-dependent: accept the dev stub or a served bundle, fail only on
    # a genuine error status.
    r = client.get("/")
    if r.status_code != 200:
        bad("root", f"{r.status_code} {r.text[:120]}")
    elif r.headers.get("content-type", "").startswith("application/json"):
        ok("GET / -> dev-mode JSON stub (no frontend bundle)")
    else:
        ok("GET / -> served frontend bundle")

    r = client.get("/api/health")
    if r.status_code != 200:
        bad("health", f"{r.status_code} {r.text[:200]}")
        return
    h = r.json()
    print(f"        status={h['status']} qdrant_reachable={h['qdrant'].get('reachable')} "
          f"weights={h['weights']}")
    if h["qdrant"].get("reachable"):
        ok("health reports Qdrant reachable")
    else:
        bad("health qdrant", h["qdrant"].get("error", "unreachable"))
    if all(h["weights"].values()):
        ok("health reports both checkpoints present")
    else:
        warn(f"missing weights: {h['weights']}")
    if h["gpu"].get("cuda_available"):
        ok(f"health reports GPU {h['gpu'].get('device_name')}")
    else:
        warn("health reports no CUDA")

    # Validation must fire before any model touches the GPU.
    cases = [
        ("empty query", "/api/search", {"query": ""}),
        ("limit over cap", "/api/search",
         {"query": "x", "limit": settings.search_limit_max + 1}),
        ("bad cloud range", "/api/search", {"query": "x", "max_cloud": 1.5}),
        ("row without col", "/api/temporal-change",
         {"t1_source": "a", "t2_source": "b", "row": 0}),
        ("bad verdict", "/api/triage", {"tile_id": "t", "verdict": "maybe"}),
        ("empty export list", "/api/export", {"tile_ids": []}),
    ]
    for name, url, body in cases:
        rr = client.post(url, json=body)
        if rr.status_code == 422:
            ok(f"422 on {name}")
        else:
            bad(f"validation:{name}", f"got {rr.status_code}, expected 422")


# ------------------------------------------------------------- 3-7 ingest/search
def check_ingest_and_search(client, scene_dir: Path) -> None:
    section("Ingest + search")

    r = client.post("/api/ingest", json={"source": str(scene_dir / "does_not_exist")})
    if r.status_code == 404:
        ok("404 on a nonexistent scene path")
    else:
        bad("ingest 404", f"got {r.status_code}")

    r = client.post("/api/ingest", json={
        "source": str(scene_dir), "scene_id": "T1_SCENE", "save_previews": True,
    })
    if r.status_code != 200:
        bad("ingest", f"{r.status_code} {r.text[:300]}")
        return
    rep = r.json()
    print(f"        {rep}")
    if rep["tiles_indexed"] == 9 and not rep["errors"]:
        ok(f"ingest indexed {rep['tiles_indexed']}/9 tiles in "
           f"{rep['elapsed_seconds']}s ({rep['tiles_per_second']}/s)")
    else:
        bad("ingest count",
            f"indexed {rep['tiles_indexed']}, errors={rep['errors'][:2]}")

    r = client.post("/api/search", json={"query": "open terrain", "limit": 5})
    if r.status_code != 200:
        bad("search", f"{r.status_code} {r.text[:300]}")
        return
    res = r.json()
    if res["count"] == 5:
        ok(f"search -> {res['count']} hits in {res['elapsed_ms']}ms, "
           f"top score {res['hits'][0]['score']:.4f}")
    else:
        bad("search count", f"got {res['count']}, expected 5")
        return

    scores = [h["score"] for h in res["hits"]]
    if scores == sorted(scores, reverse=True):
        ok("hits are score-descending")
    else:
        bad("ranking", f"unsorted: {scores}")

    top = res["hits"][0]
    if top["wgs84_bounding_box"] and len(top["wgs84_bounding_box"]) == 4:
        ok(f"hit carries bbox {top['wgs84_bounding_box']}")
    else:
        bad("hit bbox", f"got {top['wgs84_bounding_box']}")
    if top["row"] is not None and top["col"] is not None:
        ok(f"hit carries grid position r{top['row']}c{top['col']}")
    else:
        bad("hit grid", "row/col missing")
    if top["preview_url"] == f"/api/tiles/{top['tile_id']}":
        ok(f"preview_url wired: {top['preview_url']}")
    else:
        bad("preview_url", f"got {top['preview_url']}")

    # Filters
    r = client.post("/api/search", json={
        "query": "open terrain", "limit": 20, "scene_ids": ["NO_SUCH_SCENE"],
    })
    if r.status_code == 200 and r.json()["count"] == 0:
        ok("scene_ids filter on an unknown scene -> 0 hits")
    else:
        bad("scene filter", f"{r.status_code} count={r.json().get('count')}")

    r = client.post("/api/search", json={
        "query": "open terrain", "limit": 20, "scene_ids": ["T1_SCENE"],
    })
    if r.status_code == 200 and r.json()["count"] == 9:
        ok("scene_ids filter on the real scene -> all 9 hits")
    else:
        bad("scene filter real", f"count={r.json().get('count')}")

    r = client.post("/api/search", json={
        "query": "open terrain", "limit": 20, "score_threshold": 0.99,
    })
    if r.status_code == 200 and r.json()["count"] == 0:
        ok("score_threshold=0.99 -> 0 hits (CLIP scores are ~0.2)")
    else:
        bad("threshold", f"count={r.json().get('count')}")

    # Scenes listing
    r = client.get("/api/scenes")
    if r.status_code == 200 and len(r.json()) == 1 and r.json()[0]["tile_count"] == 9:
        s = r.json()[0]
        ok(f"GET /api/scenes -> {s['scene_id']} ({s['tile_count']} tiles) "
           f"bbox={s['wgs84_bounding_box']}")
    else:
        bad("scenes", f"{r.status_code} {r.text[:200]}")

    # Preview serving
    tid = top["tile_id"]
    r = client.get(f"/api/tiles/{tid}")
    if r.status_code == 200 and r.headers["content-type"] == "image/png":
        ok(f"GET /api/tiles/{tid} -> {len(r.content)} bytes PNG")
    else:
        bad("tile preview", f"{r.status_code} {r.headers.get('content-type')}")

    r = client.get("/api/tiles/NOPE_r0000c0000")
    if r.status_code == 404:
        ok("404 on an unknown tile_id")
    else:
        bad("tile 404", f"got {r.status_code}")

    # Path traversal must not escape tiles_cache.
    for evil in ("..%2f..%2fconfig_r0000c0000", "....//....//etc/passwd"):
        r = client.get(f"/api/tiles/{evil}")
        if r.status_code in (404, 400, 422):
            ok(f"traversal attempt refused ({r.status_code}) for {evil!r}")
        else:
            bad("traversal", f"{evil!r} returned {r.status_code}")


# ------------------------------------------------------------ 8-10 change
def check_change(client, t1: Path, t2: Path, offset_scene: Path) -> None:
    section("Temporal change")

    body = {
        "t1_source": str(t1), "t2_source": str(t2),
        "t1_scene_id": "T1_SCENE", "t2_scene_id": "T2_SCENE",
    }
    r = client.post("/api/temporal-change", json=body)
    if r.status_code != 200:
        bad("temporal-change", f"{r.status_code} {r.text[:300]}")
        return
    res = r.json()
    print(f"        compared={res['tiles_compared']} skipped={res['tiles_skipped']} "
          f"elapsed={res['elapsed_seconds']}s errors={res['errors'][:2]}")
    if res["tiles_compared"] == 9 and not res["errors"]:
        ok(f"compared all 9 cells in {res['elapsed_seconds']}s")
    else:
        bad("change count",
            f"compared {res['tiles_compared']}, errors={res['errors'][:2]}")

    if len(res["results"]) == 9:
        ok(f"{len(res['results'])} scored tiles returned")
    else:
        bad("change results", f"got {len(res['results'])}")
        return

    scores = [t["change_score"] for t in res["results"]]
    if scores == sorted(scores, reverse=True):
        ok(f"results sorted by change_score (max {scores[0]:.6f}, min {scores[-1]:.6f})")
    else:
        bad("change sorting", f"unsorted: {scores}")

    # The altered cell is r1c1; it should not be the quietest tile.
    altered = next((t for t in res["results"] if (t["row"], t["col"]) == (1, 1)), None)
    if altered is None:
        bad("altered cell", "r1c1 absent from results")
    else:
        rank = res["results"].index(altered) + 1
        print(f"        r1c1 (the altered cell) score={altered['change_score']:.6f} "
              f"rank {rank}/9")
        if rank <= 3:
            ok(f"altered cell ranks {rank}/9 - change signal detected")
        else:
            warn(f"altered cell ranked only {rank}/9. Synthetic noise scenes have "
                 "weak structure, so this is soft evidence, not a failure.")

    # Single-cell mode
    r = client.post("/api/temporal-change", json={**body, "row": 1, "col": 1})
    if r.status_code == 200 and r.json()["tiles_compared"] == 1:
        ok("row/col mode compares exactly one cell")
    else:
        bad("single cell", f"{r.status_code} compared={r.json().get('tiles_compared')}")

    # top_k + threshold
    r = client.post("/api/temporal-change", json={**body, "top_k": 3})
    if r.status_code == 200 and len(r.json()["results"]) == 3:
        ok("top_k=3 truncates to the 3 strongest")
    else:
        bad("top_k", f"got {len(r.json().get('results', []))}")

    r = client.post("/api/temporal-change", json={**body, "min_change_score": 0.99})
    if r.status_code == 200 and len(r.json()["results"]) == 0:
        ok("min_change_score=0.99 filters everything out")
    else:
        bad("min_change_score", f"got {len(r.json().get('results', []))}")

    # Regression guard: a genuinely uniform chip (new concrete pad, cleared
    # airstrip) must still be compared. min_stddev gates the retrieval index,
    # not change detection - see change_detect._unusable().
    flat_t2 = t2.parent / "t2_flat"
    build_scene(flat_t2, seed=7, flat_cell=(2, 2))
    r = client.post("/api/temporal-change", json={
        "t1_source": str(t1), "t2_source": str(flat_t2),
    })
    if r.status_code == 200:
        res_flat = r.json()
        cell = next((t for t in res_flat["results"]
                     if (t["row"], t["col"]) == (2, 2)), None)
        if res_flat["tiles_compared"] == 9 and cell is not None:
            ok(f"uniform chip still compared (r2c2 score "
               f"{cell['change_score']:.6f}) - flat-chip filter correctly "
               f"scoped to ingest only")
        else:
            bad("flat chip in change",
                f"compared={res_flat['tiles_compared']}, r2c2 present={cell is not None}")
    else:
        bad("flat chip request", f"{r.status_code} {r.text[:200]}")

    # Non-co-registered pair must be refused, not silently compared.
    r = client.post("/api/temporal-change", json={
        "t1_source": str(t1), "t2_source": str(offset_scene),
    })
    if r.status_code == 422 and "not comparable" in r.text:
        ok("shifted scene rejected with 422 (co-registration guard)")
    else:
        bad("coregistration guard",
            f"got {r.status_code}: {r.text[:200]}")

    # GeoJSON export
    r = client.post("/api/temporal-change/export", json={**body, "top_k": 2})
    if r.status_code != 200:
        bad("change export", f"{r.status_code} {r.text[:200]}")
        return
    gj = r.json()
    if gj.get("type") == "FeatureCollection" and len(gj["features"]) == 2:
        geom = gj["features"][0]["geometry"]
        ring = geom["coordinates"][0]
        closed = ring[0] == ring[-1]
        ok(f"export -> FeatureCollection, {len(gj['features'])} features, "
           f"{geom['type']} ring closed={closed}")
        if not closed:
            bad("geojson ring", "polygon ring is not closed")
        props = gj["features"][0]["properties"]
        if props.get("change_score") is not None and props.get("t1_scene_id"):
            ok(f"feature properties complete: {sorted(props)}")
        else:
            bad("geojson props", f"got {props}")
    else:
        bad("geojson", f"type={gj.get('type')} features={len(gj.get('features', []))}")

    if "attachment" in r.headers.get("content-disposition", ""):
        ok(f"Content-Disposition set: {r.headers['content-disposition']}")
    else:
        bad("download header", f"got {r.headers.get('content-disposition')}")


# -------------------------------------------------------- 11-12 triage/export
def check_triage(client, tile_id: str) -> None:
    section("Triage + report export")

    r = client.post("/api/triage", json={
        "tile_id": tile_id, "verdict": "confirmed",
        "query": "open terrain", "analyst_note": "matches known installation",
    })
    if r.status_code == 200 and r.json()["verdict"] == "confirmed":
        ok(f"logged confirmed verdict at {r.json()['logged_at']}")
    else:
        bad("triage post", f"{r.status_code} {r.text[:200]}")
        return

    r = client.get("/api/triage")
    if r.status_code != 200:
        bad("triage get", f"{r.status_code} {r.text[:200]}")
        return
    log = r.json()
    if log["verdicts"].get(tile_id, {}).get("verdict") == "confirmed":
        ok(f"read back: {log['stats']['tiles_triaged']} tile(s) triaged, "
           f"{log['stats']['confirmed']} confirmed")
    else:
        bad("triage readback", f"got {log['verdicts'].get(tile_id)}")

    # A later decision must supersede the earlier one.
    client.post("/api/triage", json={"tile_id": tile_id, "verdict": "false_alarm"})
    log = client.get("/api/triage").json()
    if log["verdicts"][tile_id]["verdict"] == "false_alarm":
        ok("later verdict supersedes earlier (append-only, last wins)")
    else:
        bad("verdict supersede", f"got {log['verdicts'][tile_id]['verdict']}")
    if log["stats"]["total_entries"] == 2 and log["stats"]["tiles_triaged"] == 1:
        ok("2 log entries collapse to 1 triaged tile")
    else:
        bad("audit stats", f"got {log['stats']}")

    # Report export
    r = client.post("/api/export", json={
        "tile_ids": [tile_id, "GHOST_r9999c9999"], "query": "open terrain",
    })
    if r.status_code != 200:
        bad("export", f"{r.status_code} {r.text[:200]}")
        return
    gj = r.json()
    if len(gj["features"]) == 1 and gj["features"][0]["properties"]["verdict"] == "false_alarm":
        ok(f"export -> 1 feature, verdict={gj['features'][0]['properties']['verdict']}, "
           f"note preserved={bool(gj['features'][0]['properties']['analyst_note'])}")
    else:
        bad("export features", f"got {len(gj['features'])} features")
    if gj.get("missing_tile_ids") == ["GHOST_r9999c9999"]:
        ok("unknown tile reported in missing_tile_ids rather than silently dropped")
    else:
        bad("export missing", f"got {gj.get('missing_tile_ids')}")

    r = client.post("/api/export", json={"tile_ids": ["GHOST_r0000c0000"]})
    if r.status_code == 404:
        ok("404 when nothing is exportable")
    else:
        bad("export 404", f"got {r.status_code}")


# -------------------------------------------------------------------- runner
def main() -> int:
    print("=" * 68)
    print(" TRINETRA AI - PHASE 4 API VERIFICATION")
    print("=" * 68)

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"{FAIL} fastapi.testclient unavailable: {exc}")
        print("       pip install httpx==0.28.1")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="trinetra_phase4_"))

    # Redirect every persistent path at the throwaway tree BEFORE the app boots.
    from app.services.vector_store import VectorStore, vector_store
    orig = (vector_store.collection, settings.tiles_cache_dir,
            settings.audit_log_path, settings.sample_data_dir)
    vector_store.collection = TEST_COLLECTION
    settings.tiles_cache_dir = tmp / "tiles_cache"
    settings.audit_log_path = tmp / "audit_log.jsonl"
    settings.sample_data_dir = tmp

    try:
        t1 = tmp / "t1"
        t2 = tmp / "t2"
        shifted = tmp / "shifted"
        build_scene(t1, seed=7)
        build_scene(t2, seed=7, alter_cell=(1, 1))
        # Same size/CRS, origin moved 5 km east - must fail co-registration.
        build_scene(shifted, seed=7, origin=(ORIGIN_X + 5000.0, ORIGIN_Y))
        ok(f"built 3 synthetic scenes under {tmp.name}")

        from app.main import app

        with TestClient(app) as client:
            for fn, args in (
                (check_basics, (client,)),
                (check_ingest_and_search, (client, t1)),
                (check_change, (client, t1, t2, shifted)),
            ):
                try:
                    fn(*args)
                except Exception as exc:
                    bad(fn.__name__, f"unhandled {type(exc).__name__}: {exc}")
                    traceback.print_exc()
            try:
                check_triage(client, "T1_SCENE_r0000c0000")
            except Exception as exc:
                bad("check_triage", f"unhandled {type(exc).__name__}: {exc}")
                traceback.print_exc()
    finally:
        try:
            store = VectorStore(collection=TEST_COLLECTION)
            if store.client.collection_exists(TEST_COLLECTION):
                store.client.delete_collection(TEST_COLLECTION)
                ok("test collection dropped")
            store.close()
        except Exception as exc:
            warn(f"cleanup failed: {exc}")
        vector_store.close()
        (vector_store.collection, settings.tiles_cache_dir,
         settings.audit_log_path, settings.sample_data_dir) = orig
        shutil.rmtree(tmp, ignore_errors=True)

    section("Summary")
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed: {', '.join(_failures)}")
    else:
        print(f"{PASS} Phase 4 verified.")
    if _warnings:
        print(f"{WARN} {len(_warnings)} warning(s) - non-blocking.")
    print("=" * 68)
    return 1 if _failures else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
