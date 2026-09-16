"""
Trinetra AI - Phase 4: HTTP surface.

Concurrency note: RemoteCLIP and Prithvi share one 8 GB GPU and neither
encoder serializes its own forward pass. FastAPI runs `def` handlers in a
threadpool, so two simultaneous ingests would race for VRAM. _GPU_LOCK makes
every GPU-heavy route mutually exclusive; a second caller waits instead of
triggering an OOM. Text search holds it too, but only for a single tokenized
forward pass (sub-millisecond), so it is not a throughput concern.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.schemas.api import (
    ExportRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    DatasetSummary,
    PipelineIngestRequest,
    PipelineIngestResponse,
    SceneSummary,
    SearchHitModel,
    SearchRequest,
    SearchResponse,
    TemporalChangeRequest,
    TemporalChangeResponse,
    TriageRequest,
    TriageResponse,
)
from app.services import audit
from app.services.change_detect import compare_scenes, to_geojson
from app.services.ingest import ingest_scene
from app.services.ml_inference import InferenceError, remoteclip, vram_report
from app.services.raster_engine import RasterEngineError
from app.services.vector_store import VectorStoreError, vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

_GPU_LOCK = threading.Lock()
_TILE_ID_RE = re.compile(r"^(?P<scene>.+)_r(?P<row>\d{4})c(?P<col>\d{4})$")


# ------------------------------------------------------------------ helpers
def _split_tile_id(tile_id: str) -> tuple[str, int, int] | None:
    m = _TILE_ID_RE.match(tile_id)
    if not m:
        return None
    return m.group("scene"), int(m.group("row")), int(m.group("col"))


def _preview_path(tile_id: str) -> Path | None:
    """
    Resolve a tile_id to its cached PNG, refusing anything outside tiles_cache.

    tile_ids reach us from the client, so a crafted '../..' value must not be
    able to read arbitrary files off the host.
    """
    parts = _split_tile_id(tile_id)
    if parts is None:
        return None
    scene_id, _, _ = parts
    root = settings.tiles_cache_dir.resolve()
    candidate = (root / scene_id / f"{tile_id}.png").resolve()
    if root not in candidate.parents:
        logger.warning("Rejected preview path traversal attempt: %r", tile_id)
        return None
    return candidate if candidate.is_file() else None


def _hit_to_model(hit) -> SearchHitModel:
    parts = _split_tile_id(hit.tile_id)
    row = hit.payload.get("row")
    col = hit.payload.get("col")
    if row is None and parts:
        _, row, col = parts
    return SearchHitModel(
        tile_id=hit.tile_id,
        score=round(hit.score, 6),
        scene_id=hit.scene_id,
        acquisition_timestamp=hit.acquisition_timestamp,
        wgs84_bounding_box=hit.wgs84_bounding_box,
        scl_cloud_coverage=hit.scl_cloud_coverage,
        row=row,
        col=col,
        preview_url=(f"/api/tiles/{hit.tile_id}"
                     if _preview_path(hit.tile_id) else None),
    )


# ------------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse, tags=["diagnostics"])
def health() -> HealthResponse:
    """Liveness plus a real dependency check - never cached."""
    qdrant = vector_store.health()
    gpu = vram_report()
    weights = {
        "remoteclip": settings.remoteclip_checkpoint.is_file(),
        "prithvi": settings.prithvi_checkpoint.is_file(),
    }
    degraded = not qdrant.get("reachable") or not all(weights.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.api_version,
        qdrant=qdrant,
        gpu=gpu,
        weights=weights,
    )


# ------------------------------------------------------------------- ingest
@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Tile a scene, embed every surviving chip, index it in Qdrant.

    Synchronous: a full Sentinel-2 tile is ~1800 chips and takes minutes. The
    connection stays open for the duration; a background-job queue is Phase 6
    work, not something to fake with a fire-and-forget thread here.
    """
    source = Path(req.source).expanduser()
    if not source.is_absolute():
        source = (settings.sample_data_dir / req.source).resolve()
    if not source.exists():
        raise HTTPException(404, f"Scene source not found: {source}")

    with _GPU_LOCK:
        try:
            report = ingest_scene(
                str(source),
                scene_id=req.scene_id,
                max_tiles=req.max_tiles,
                replace=req.replace,
                save_previews=req.save_previews,
            )
        except RasterEngineError as exc:
            raise HTTPException(422, f"Scene cannot be read: {exc}") from exc
        except (InferenceError, VectorStoreError) as exc:
            raise HTTPException(503, str(exc)) from exc

    return IngestResponse(**report.as_dict())


# ------------------------------------------------------------------- search
from app.core.state import state


@router.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """Semantic vector search against the indexed Sentinel-2 chips."""
    state.mark_search()
    started = time.perf_counter()
    try:
        with _GPU_LOCK:
            qvec = remoteclip.encode_text(req.query)[0]
    except InferenceError as exc:
        raise HTTPException(503, f"Text encoder unavailable: {exc}") from exc

    try:
        raw_hits = vector_store.search(
            qvec,
            limit=req.limit * 3 if req.bounding_box else req.limit, # Fetch more to account for post-filtering
            score_threshold=req.score_threshold,
            scene_ids=req.scene_ids,
            max_cloud=req.max_cloud,
            date_range=req.date_range,
        )
    except VectorStoreError as exc:
        raise HTTPException(503, f"Vector store error: {exc}") from exc

    hits = []
    for h in raw_hits:
        if req.bounding_box and h.wgs84_bounding_box:
            min_lon, min_lat, max_lon, max_lat = req.bounding_box
            hw, hs, he, hn = h.wgs84_bounding_box
            # Check intersection
            if not (he < min_lon or hw > max_lon or hn < min_lat or hs > max_lat):
                hits.append(h)
        else:
            hits.append(h)
        if len(hits) == req.limit:
            break

    return SearchResponse(
        query=req.query,
        count=len(hits),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        hits=[_hit_to_model(h) for h in hits],
    )


@router.get("/scenes", response_model=list[SceneSummary], tags=["search"])
def scenes() -> list[SceneSummary]:
    """Indexed scenes, for the frontend's filter dropdown and initial map extent."""
    try:
        vector_store.ensure_collection()
        return [SceneSummary(**s) for s in vector_store.list_scenes()]
    except VectorStoreError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/tiles/{tile_id}", tags=["search"])
def tile_preview(tile_id: str) -> FileResponse:
    """Cached PNG chip. Immutable once written, so it caches aggressively."""
    path = _preview_path(tile_id)
    if path is None:
        raise HTTPException(404, f"No cached preview for {tile_id}")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/datasets", response_model=list[DatasetSummary], tags=["datasets"])
def list_datasets() -> list[DatasetSummary]:
    """Scans local drop zones and sample data for available .SAFE folders or .tif files."""
    datasets = []
    
    def scan_dir(dir_path: Path, location_name: str):
        if not dir_path.exists():
            return
        
        for item in dir_path.iterdir():
            if item.name == ".gitkeep":
                continue
                
            if item.is_dir() and ("_L2A" in item.name or ".SAFE" in item.name or "S2" in item.name or "DEMO" in item.name):
                # Try to extract date like 20230105
                date_match = re.search(r"_(20\d{6})T", item.name)
                date_str = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:8]}" if date_match else None
                datasets.append(DatasetSummary(name=item.name, path=str(item), date=date_str, location=location_name))
            elif item.is_file() and item.suffix.lower() in [".tif", ".tiff"]:
                # Try to extract date
                date_match = re.search(r"_(20\d{6})", item.name)
                date_str = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:8]}" if date_match else None
                datasets.append(DatasetSummary(name=item.name, path=str(item), date=date_str, location=location_name))
            # Also support generic folders starting with scene_
            elif item.is_dir() and item.name.startswith("scene_"):
                date_match = re.search(r"scene_(20\d{2})_(\d{2})", item.name)
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-01" if date_match else None
                datasets.append(DatasetSummary(name=item.name, path=str(item), date=date_str, location=location_name))

    scan_dir(settings.secure_drop_zone_dir, "secure_drop_zone")
    scan_dir(settings.sample_data_dir, "sample_data")
    
    return sorted(datasets, key=lambda x: x.date or "", reverse=True)


# --------------------------------------------------------- temporal change
@router.post("/temporal-change", response_model=TemporalChangeResponse,
             tags=["change"])
def temporal_change(req: TemporalChangeRequest) -> TemporalChangeResponse:
    """
    Prithvi change scores between two co-registered acquisitions.

    Nothing is persisted - features are computed, diffed, and discarded.
    """
    def _resolve(raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (settings.sample_data_dir / raw).resolve()
        if not p.exists():
            raise HTTPException(404, f"Scene source not found: {p}")
        return p

    t1, t2 = _resolve(req.t1_source), _resolve(req.t2_source)

    with _GPU_LOCK:
        try:
            report = compare_scenes(
                str(t1), str(t2),
                t1_scene_id=req.t1_scene_id,
                t2_scene_id=req.t2_scene_id,
                row=req.row,
                col=req.col,
                min_change_score=req.min_change_score,
                max_tiles=req.max_tiles,
                top_k=req.top_k,
                skip_cloudy=req.skip_cloudy,
            )
        except RasterEngineError as exc:
            raise HTTPException(422, str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(503, str(exc)) from exc

    return TemporalChangeResponse(**report.as_dict())


@router.post("/temporal-change/export", tags=["change"])
def temporal_change_export(req: TemporalChangeRequest) -> Response:
    """Same comparison, returned as a downloadable GeoJSON FeatureCollection."""
    result = temporal_change(req)
    from app.services.change_detect import ChangeReport, ChangeTile

    report = ChangeReport(
        t1_scene_id=result.t1_scene_id,
        t2_scene_id=result.t2_scene_id,
        t1_timestamp=result.t1_timestamp,
        t2_timestamp=result.t2_timestamp,
        results=[ChangeTile(**t.model_dump()) for t in result.results],
    )
    name = f"trinetra_change_{result.t1_scene_id}_vs_{result.t2_scene_id}.geojson"
    return JSONResponse(
        content=to_geojson(report),
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
        media_type="application/geo+json",
    )


# ---------------------------------------------------------------- triage
@router.post("/triage", response_model=TriageResponse, tags=["triage"])
def triage(req: TriageRequest) -> TriageResponse:
    """Record an analyst's Confirm Intel / False Alarm decision."""
    try:
        entry = audit.record(
            req.tile_id, req.verdict,
            query=req.query, analyst_note=req.analyst_note,
        )
    except audit.AuditError as exc:
        raise HTTPException(500, str(exc)) from exc
    return TriageResponse(
        tile_id=entry["tile_id"],
        verdict=entry["verdict"],
        logged_at=entry["logged_at"],
        total_entries=audit.count(),
    )

import shutil

@router.post("/triage/false_alarm", response_model=TriageResponse, tags=["triage"])
def triage_false_alarm(req: TriageRequest) -> TriageResponse:
    """Record a false alarm and move the patch to training_data/."""
    try:
        # First record the verdict
        req.verdict = "False Alarm"
        entry = audit.record(
            req.tile_id, req.verdict,
            query=req.query, analyst_note=req.analyst_note,
        )
        
        # Now move the file
        parts = _split_tile_id(req.tile_id)
        if parts:
            scene, _, _ = parts
            src_png = settings.tiles_cache_dir / scene / f"{req.tile_id}.png"
            if src_png.exists():
                settings.training_data_dir.mkdir(parents=True, exist_ok=True)
                dst_png = settings.training_data_dir / f"{req.tile_id}.png"
                shutil.copy2(src_png, dst_png)
                
    except audit.AuditError as exc:
        raise HTTPException(500, str(exc)) from exc
    return TriageResponse(
        tile_id=entry["tile_id"],
        verdict=entry["verdict"],
        logged_at=entry["logged_at"],
        total_entries=audit.count(),
    )


@router.get("/triage", tags=["triage"])
def triage_log(
    tile_ids: list[str] | None = Query(None, description="Filter to these tiles."),
) -> dict:
    """Latest verdict per tile, plus aggregate counts."""
    try:
        verdicts = audit.latest_verdicts()
        summary = audit.stats()
    except audit.AuditError as exc:
        raise HTTPException(500, str(exc)) from exc
    if tile_ids:
        verdicts = {k: v for k, v in verdicts.items() if k in set(tile_ids)}
    return {"stats": summary, "verdicts": verdicts}


@router.post("/export", tags=["triage"])
def export_report(req: ExportRequest) -> Response:
    """
    GeoJSON of the analyst's selected tiles, annotated with triage verdicts.

    Geometry comes from Qdrant payloads, so only indexed tiles can be exported.
    """
    try:
        verdicts = audit.latest_verdicts()
    except audit.AuditError as exc:
        raise HTTPException(500, str(exc)) from exc

    features: list[dict] = []
    missing: list[str] = []
    for tile_id in req.tile_ids:
        verdict = verdicts.get(tile_id, {}).get("verdict")
        if verdict is None and not req.include_unverified:
            continue
        try:
            payload = vector_store.get_tile(tile_id)
        except VectorStoreError as exc:
            raise HTTPException(503, str(exc)) from exc
        if not payload:
            missing.append(tile_id)
            continue
        bbox = payload.get("wgs84_bounding_box")
        if not bbox or len(bbox) != 4:
            missing.append(tile_id)
            continue
        w, s, e, n = bbox
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            },
            "properties": {
                "tile_id": tile_id,
                "scene_id": payload.get("scene_id"),
                "acquisition_timestamp": payload.get("acquisition_timestamp"),
                "scl_cloud_coverage": payload.get("scl_cloud_coverage"),
                "verdict": verdict or "unverified",
                "analyst_note": verdicts.get(tile_id, {}).get("analyst_note"),
                "query": req.query,
            },
        })

    if not features:
        raise HTTPException(404, "No exportable tiles - none were found in the index.")

    collection = {
        "type": "FeatureCollection",
        "name": "trinetra_intel_report",
        "features": features,
    }
    if missing:
        collection["missing_tile_ids"] = missing
    return JSONResponse(
        content=collection,
        headers={"Content-Disposition": 'attachment; filename="trinetra_report.geojson"'},
        media_type="application/geo+json",
    )

# --------------------------------------------------------- pipeline ingest
import uuid
import json as _json

_PIPELINE_JOBS: dict[str, dict] = {}


@router.get("/pipeline/bbox", tags=["pipeline"])
def pipeline_bbox() -> dict:
    """
    Extract the AOI bounding box from the most recent manifest.json in the
    secure drop zone. Returns the bbox so the frontend can auto-fill coordinates
    for new downloads without manual entry.
    """
    best_bbox = None
    best_date = ""
    for scan_dir in (settings.secure_drop_zone_dir, settings.sample_data_dir):
        if not scan_dir.exists():
            continue
        for scene_dir in scan_dir.iterdir():
            manifest = scene_dir / "manifest.json" if scene_dir.is_dir() else None
            if manifest and manifest.is_file():
                try:
                    data = _json.loads(manifest.read_text(encoding="utf-8"))
                    bbox = data.get("aoi_bbox_wgs84")
                    acquired = data.get("acquired", "")
                    if bbox and len(bbox) == 4:
                        if acquired > best_date:
                            best_bbox = bbox
                            best_date = acquired
                except Exception:
                    continue
    if best_bbox is None:
        # Fallback to Jaisalmer default
        best_bbox = [70.8, 26.8, 71.0, 27.0]
    return {"bbox": best_bbox}


@router.post("/pipeline/ingest", response_model=PipelineIngestResponse,
             status_code=202, tags=["pipeline"])
def pipeline_ingest(req: PipelineIngestRequest) -> PipelineIngestResponse:
    """
    Launch ingest_pipeline.py in a background thread to download Sentinel-2
    scenes from Element84 into the secure drop zone.
    """
    # Prevent concurrent downloads
    for jid, job in _PIPELINE_JOBS.items():
        if job["status"] == "running":
            return PipelineIngestResponse(
                status="running",
                message=f"A pipeline job is already running (job {jid}). Wait for it to finish.",
                job_id=jid,
            )

    job_id = uuid.uuid4().hex[:12]
    _PIPELINE_JOBS[job_id] = {
        "status": "running",
        "message": "Pipeline started",
        "logs": [],
    }

    def _run():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from ingest_pipeline import main as pipeline_main

            argv = [
                "--bbox", *[str(v) for v in req.bbox],
                "--start", req.start_date,
                "--end", req.end_date,
                "--max-cloud", str(req.max_cloud),
                "--limit", str(req.limit),
                "--out", str(settings.secure_drop_zone_dir),
            ]
            logger.info("Pipeline job %s started: %s", job_id, " ".join(argv))
            _PIPELINE_JOBS[job_id]["message"] = f"Downloading (args: {' '.join(argv)})"
            rc = pipeline_main(argv)
            if rc == 0:
                _PIPELINE_JOBS[job_id]["status"] = "completed"
                _PIPELINE_JOBS[job_id]["message"] = "Download completed successfully."
            else:
                _PIPELINE_JOBS[job_id]["status"] = "failed"
                _PIPELINE_JOBS[job_id]["message"] = f"Pipeline exited with code {rc}."
            logger.info("Pipeline job %s finished with rc=%d", job_id, rc)
        except Exception as exc:
            _PIPELINE_JOBS[job_id]["status"] = "failed"
            _PIPELINE_JOBS[job_id]["message"] = f"Pipeline error: {exc}"
            logger.exception("Pipeline job %s failed", job_id)

    t = threading.Thread(target=_run, name=f"pipeline-{job_id}", daemon=True)
    t.start()

    return PipelineIngestResponse(
        status="accepted",
        message="Pipeline job started. Poll /api/pipeline/status/{job_id} for progress.",
        job_id=job_id,
    )


@router.get("/pipeline/status/{job_id}", response_model=PipelineIngestResponse,
            tags=["pipeline"])
def pipeline_status(job_id: str) -> PipelineIngestResponse:
    """Check the status of a pipeline download job."""
    job = _PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"No pipeline job with id {job_id}")
    return PipelineIngestResponse(
        status=job["status"],
        message=job["message"],
        job_id=job_id,
    )


# ------------------------------------------------------------- watchdog stream
import asyncio
from fastapi.responses import StreamingResponse
from app.core.state import broadcaster

async def event_generator(q: asyncio.Queue):
    try:
        while True:
            message = await q.get()
            yield f"data: {message}\n\n"
    except asyncio.CancelledError:
        pass

@router.get("/watchdog/stream", tags=["watchdog"])
async def watchdog_stream():
    """SSE endpoint for watchdog alerts."""
    q = broadcaster.subscribe()
    async def sse_wrapper():
        try:
            async for msg in event_generator(q):
                yield msg
        finally:
            broadcaster.unsubscribe(q)
    return StreamingResponse(sse_wrapper(), media_type="text/event-stream")
