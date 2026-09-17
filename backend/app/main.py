"""
Trinetra AI - Phase 4: FastAPI application.

Run from the repo root:
    .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend --reload

Models are NOT preloaded at startup. Both encoders are lazy singletons, so the
API answers /health in milliseconds on a cold start and only pays the ~3 s load
cost on the first request that actually needs a GPU.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.services.vector_store import VectorStoreError, vector_store

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("trinetra")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.tiles_cache_dir.mkdir(parents=True, exist_ok=True)
    # The audit log's directory may be a fresh bind mount with nothing in it.
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    import asyncio
    from app.core.state import broadcaster
    from app.services.watchdog_service import start_watchdog
    
    # Bind the running loop so the watchdog's worker thread can push SSE alerts
    # into it. This assignment previously set `broadcaster.loop`, which nothing
    # read - broadcast() called get_running_loop() from the worker thread, hit
    # RuntimeError, and silently discarded every alert.
    broadcaster.set_loop(asyncio.get_running_loop())

    # Guarded, matching the Qdrant policy below. This call was previously bare,
    # so ANY failure inside it - an unwritable bind mount, a PollingObserver
    # thread that will not spawn - propagated out of lifespan, uvicorn printed
    # "Application startup failed. Exiting." and the process died. From the
    # outside that looks like `curl: (52) Empty reply from server`: the port is
    # published, so the TCP connect succeeds, but nothing is listening behind it.
    # The watchdog is an auxiliary feature; it must not be able to take down
    # search, change detection, or /health.
    observer = None
    try:
        observer = start_watchdog()
        logger.info("AOI Watchdog started.")
    except Exception:
        logger.exception(
            "Watchdog failed to start - continuing WITHOUT drop-zone monitoring. "
            "Sneakernet ingest via POST /api/ingest is unaffected."
        )

    # Bootstrap the collection now so the first /search does not race a create.
    # A missing Qdrant is logged, not fatal: /health should still answer and
    # report the outage rather than the whole API refusing to start.
    try:
        vector_store.ensure_collection()
        logger.info("Qdrant ready: %s", vector_store.health())
    except VectorStoreError as exc:
        logger.error("Qdrant unavailable at startup: %s", exc)

    for name, path in (("RemoteCLIP", settings.remoteclip_checkpoint),
                       ("Prithvi", settings.prithvi_checkpoint)):
        if not path.is_file():
            logger.warning("%s weights missing at %s - related routes will 503.",
                           name, path)

    logger.info("%s %s ready (models load on first use)",
                settings.api_title, settings.api_version)
    yield

    # Stop the observer before closing Qdrant. Previously it was started and
    # never stopped, so its polling threads outlived every --reload cycle.
    if observer is not None:
        try:
            observer.stop()
            observer.join(timeout=10)
        except Exception:
            logger.exception("Watchdog shutdown raised (ignored)")

    vector_store.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Offline semantic retrieval and multi-temporal change analysis "
                "for satellite imagery.",
    lifespan=lifespan,
)

# Air-gapped deployment: explicit origins only, no wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback server-side; return a generic message to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See server logs for details."},
    )


@app.get("/api", tags=["diagnostics"])
def api_root() -> dict:
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "endpoints": [
            "POST /api/ingest",
            "POST /api/search",
            "POST /api/temporal-change",
            "POST /api/temporal-change/export",
            "POST /api/triage",
            "GET  /api/triage",
            "POST /api/export",
            "GET  /api/scenes",
            "GET  /api/tiles/{tile_id}",
            "GET  /api/health",
        ],
    }


# The static mount is registered LAST and deliberately. A mount at "/" is a
# catch-all, so anything declared after it becomes unreachable - /api/*, /docs
# and /openapi.json all resolve first only because they already exist here.
#
# html=True makes StaticFiles serve index.html for "/" and fall back to it for
# unknown paths, which is what a client-routed SPA needs.
if settings.frontend_dist_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=settings.frontend_dist_dir, html=True),
        name="frontend",
    )
    logger.info("Serving built frontend from %s", settings.frontend_dist_dir)
else:
    # Local dev: Vite owns :5173 and proxies /api here, so there is no bundle to
    # serve. Keep "/" informative rather than a bare 404.
    logger.info(
        "No frontend bundle at %s - serving API only (use the Vite dev server).",
        settings.frontend_dist_dir,
    )

    @app.get("/", tags=["diagnostics"])
    def root() -> dict:
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "frontend": "not built - run `npm run dev` in frontend/, or build the Docker image",
            "api": "/api",
            "docs": "/docs",
        }
