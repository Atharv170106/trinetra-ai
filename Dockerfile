# Trinetra AI - single-image build.
#
# Stage 1 compiles the React bundle with Node; stage 2 is the CUDA runtime that
# serves both the API and that bundle. One image, one origin: no nginx, no CORS,
# no dev proxy, and one artifact to sideload onto an air-gapped host.
#
# The BUILD needs network access (pip + npm). The RUNTIME does not - that is the
# whole point. Build once on a connected machine, then:
#     docker save trinetra-ai:latest -o trinetra-ai.tar
#     docker load -i trinetra-ai.tar          # on the air-gapped host
#
# Model weights are NOT baked in. They are ~1.5 GB and change independently of
# the code, so they are bind-mounted read-only (see docker-compose.yml).

# An ARG referenced by a FROM must be declared BEFORE the first FROM - args
# declared inside a stage are scoped to that stage and read as blank up here.
# Override with: docker compose build --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# --------------------------------------------------------------- stage 1: web
FROM node:20-bookworm-slim AS web

WORKDIR /build

# Copy manifests alone first so a source-only edit does not invalidate the
# dependency layer.
COPY frontend/package.json frontend/package-lock.json ./
# `npm ci` installs exactly the lockfile - never resolves fresh versions, which
# also means the build is reproducible on an internal registry mirror.
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build && test -f dist/index.html

# ------------------------------------------------------------ stage 2: server
# Torch + CUDA runtime is pre-built here; installing torch via pip instead would
# add ~2.5 GB of download to every build. Ships Python 3.12 (via Miniconda) -
# one minor ahead of the 3.11 venv, so treat Docker-only bugs with suspicion.
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hard offline guarantee: if any code path ever tries to reach the Hub, it
    # fails loudly at load time instead of silently hanging on a dead network.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Keep GDAL's block cache small; 16 GB hosts cannot afford the default.
    GDAL_CACHEMAX=256 \
    GDAL_NUM_THREADS=2 \
    # rasterio wheels bundle their own PROJ data - do not let a stale host var win
    PROJ_NETWORK=OFF

WORKDIR /app

# requirements.txt deliberately omits torch (already in the base image), so this
# will not pull a second CUDA wheel set. numpy is pinned <2.0 and may DOWNGRADE
# the base image's numpy; that is intended - rasterio + open_clip want 1.26.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    "pystac-client==0.9.0" "pystac>=1.11.0" \
 && python -c "import rasterio, torch, open_clip; \
from importlib.metadata import version; \
print('rasterio', rasterio.__version__, '| torch', torch.__version__, \
'| qdrant', version('qdrant-client'), '| open_clip', version('open_clip_torch'))"

COPY backend/app ./backend/app
COPY backend/ingest_pipeline.py ./backend/ingest_pipeline.py
COPY --from=web /build/dist ./frontend/dist

# Writable state lives under /data (bind-mounted from the host by compose),
# never in the image tree. Both directories are created here so the container
# starts cleanly even on a checkout where the host dirs do not exist yet.
RUN mkdir -p /data/tiles_cache /data/audit \
 && useradd --create-home --uid 1000 trinetra \
 && chown -R trinetra:trinetra /data

ENV TRINETRA_TILES_CACHE_DIR=/data/tiles_cache \
    TRINETRA_AUDIT_LOG_PATH=/data/audit/audit_log.jsonl \
    TRINETRA_MODEL_WEIGHTS_DIR=/app/backend/model_weights \
    TRINETRA_SAMPLE_DATA_DIR=/app/backend/sample_data \
    TRINETRA_QDRANT_URL=http://qdrant:6333

USER trinetra
WORKDIR /app/backend
EXPOSE 8000

# No curl in this base image, so the probe is plain stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# Exactly one worker. The GPU serialization lock in routes.py is process-local,
# so a second worker would let two requests contend for 8 GB of VRAM and OOM.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
