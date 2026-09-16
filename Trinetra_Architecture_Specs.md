# Trinetra AI — System Architecture & Project Specification Document
**Competition:** Smart India Hackathon (SIH) 2026 | **Problem Statement ID:** 26227  
**Organization:** Ministry of Defence (MoD) / Indian Army (DGIS)  
**Title:** Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery  
**Theme:** Space Technology | **Category:** Software

---

## 1. Executive Summary & Core Objective

Trinetra AI is a **sovereign, air-gapped, multimodal satellite intelligence search engine** designed for defense and intelligence analysts. It replaces manual, coordinate-based satellite image scanning with **natural-language semantic search**, **automated area-of-interest monitoring**, and **multi-temporal change detection** — all running 100% offline on consumer-grade hardware.

The system ingests Copernicus Sentinel-2 L2A Cloud-Optimized GeoTIFFs (COGs), tiles them into spatial chips, extracts high-dimensional semantic embeddings via dual AI models (**RemoteCLIP** for visual-language retrieval, **Prithvi-EO-2.0** for temporal change analysis), and indexes them in a local vector database. Analysts interact with the data through an interactive React map dashboard featuring an **Omni-Search bar**, **spatial drawing tools**, **real-time watchdog alerts**, and a **triage workflow**.

---

## 2. Hardware Constraints & Deployment Rules

| Resource | Specification | Design Implication |
|----------|--------------|-------------------|
| **GPU** | NVIDIA RTX 4060 Laptop (8 GB VRAM) | Both models reside in FP16; a `_GPU_LOCK` mutex prevents concurrent inference OOM |
| **System RAM** | 16 GB DDR5 | Windowed reads via `rasterio` only — full scenes never loaded into memory |
| **Storage** | SSD recommended | Qdrant vector volume + tile PNG cache are I/O intensive |
| **Network** | **Air-gapped** (zero external connectivity) | No cloud APIs, CDNs, or remote model endpoints. All weights pre-downloaded and mounted read-only |

---

## 3. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANALYST WORKSTATION                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    React + Vite Frontend                          │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │  │
│  │  │ Omni-Search │  │ Leaflet Map  │  │    Triage Modal         │  │  │
│  │  │ Bar (text,  │  │ + Geoman     │  │  (SSE Watchdog Alerts)  │  │  │
│  │  │ coords,     │  │ Drawing      │  │  Acknowledge / False    │  │  │
│  │  │ date range) │  │ Tools        │  │  Alarm                  │  │  │
│  │  └──────┬──────┘  └──────┬───────┘  └───────────┬─────────────┘  │  │
│  │         │                │                      │                 │  │
│  │         └────────────────┼──────────────────────┘                 │  │
│  │                          │ HTTP / SSE                             │  │
│  └──────────────────────────┼───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                  FastAPI Backend (:8080)                          │  │
│  │                          │                                        │  │
│  │  ┌──────────────────────────────────────────────────────────┐    │  │
│  │  │                    API Router (routes.py)                 │    │  │
│  │  │  /search  /ingest  /temporal-change  /triage              │    │  │
│  │  │  /triage/false_alarm  /watchdog/stream  /health           │    │  │
│  │  └────────────┬──────────────────────────────┬──────────────┘    │  │
│  │               │                              │                    │  │
│  │  ┌────────────▼────────────┐  ┌──────────────▼──────────────┐    │  │
│  │  │   ML Inference Engine   │  │   Raster Engine             │    │  │
│  │  │  ┌──────────────────┐   │  │  SceneReader (windowed I/O) │    │  │
│  │  │  │ RemoteCLIP       │   │  │  Tiling + SCL Cloud Mask    │    │  │
│  │  │  │ (ViT-B-32, FP16) │   │  │  WGS84 Bounding Box Proj.  │    │  │
│  │  │  ├──────────────────┤   │  │  Multi-resolution Upsample  │    │  │
│  │  │  │ Prithvi-EO-2.0   │   │  └────────────────────────────┘    │  │
│  │  │  │ (300M, FP16)     │   │                                     │  │
│  │  │  └──────────────────┘   │  ┌─────────────────────────────┐    │  │
│  │  │  _GPU_LOCK (mutex)      │  │   Watchdog Service          │    │  │
│  │  └─────────────────────────┘  │  Monitors secure_drop_zone/ │    │  │
│  │               │               │  30s Priority Queue pause    │    │  │
│  │               │               │  SSE EventBroadcaster        │    │  │
│  │               ▼               └─────────────────────────────┘    │  │
│  │  ┌─────────────────────────┐                                     │  │
│  │  │   Qdrant Vector Store   │  ┌─────────────────────────────┐    │  │
│  │  │  512-dim Cosine Index   │  │  Audit Service              │    │  │
│  │  │  Payload Indexes:       │  │  JSONL verdict logging      │    │  │
│  │  │   scene_id (keyword)    │  │  False alarm → training_    │    │  │
│  │  │   acquisition_timestamp │  │  data/ quarantine           │    │  │
│  │  │   scl_cloud_coverage    │  └─────────────────────────────┘    │  │
│  │  └─────────────────────────┘                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Docker Compose                                │  │
│  │  trinetra (FastAPI + React)  ◄──►  qdrant (v1.16, named vol.)   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────┐
         │     STAGING MACHINE (internet-connected)     │
         │                                              │
         │  ingest_pipeline.py                          │
         │  pystac_client → Earth Search STAC API       │
         │  Downloads COG bands → secure_drop_zone/     │
         │                                              │
         │  ──── sneakernet / USB ────►  Air-gapped     │
         └─────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic | REST API, request validation, lifecycle |
| **Geospatial** | GDAL, rasterio, shapely, pyproj, NumPy, Pillow | Windowed COG reads, CRS projection, tiling |
| **Semantic Search** | RemoteCLIP (ViT-B-32) via `open_clip_torch`, CUDA FP16 | 512-dim text/image embedding, cosine similarity |
| **Change Detection** | Prithvi-EO-2.0-300M via HuggingFace Transformers, CUDA FP16 | 6-band multispectral temporal feature comparison |
| **Vector Database** | Qdrant v1.16 (Dockerized, persistent named volume) | Cosine ANN search, payload filtering, on-disk vectors |
| **Background Service** | Python `watchdog` library | File system monitoring of `secure_drop_zone/` |
| **Real-Time Alerts** | Server-Sent Events (SSE) via `EventBroadcaster` | Push watchdog detections to frontend without polling |
| **Frontend** | React 18 (Vite), Leaflet.js, Leaflet-Geoman | Omni-Search, spatial drawing, triage modal |
| **Data Acquisition** | `pystac_client` + `rasterio` (staging-only script) | STAC catalogue search, windowed HTTP range reads |
| **Infrastructure** | Docker & Docker Compose, NVIDIA Container Toolkit | GPU passthrough, unified build, air-gapped deployment |

---

## 5. Core Processing Pipeline

### 5.1 Secure Ingestion Pipeline (Online Staging)
1. Analyst runs `ingest_pipeline.py` on an internet-connected staging machine.
2. Script queries Earth Search STAC API for Sentinel-2 L2A scenes matching a bounding box + date range.
3. Required bands (B02, B03, B04, B8A, B11, B12, SCL) are downloaded as windowed COG crops.
4. Output is a self-contained directory of per-band GeoTIFFs dropped into `secure_drop_zone/`.
5. Directory is transferred to the air-gapped host via sneakernet (USB, secure media).

### 5.2 Air-Gapped AOI Watchdog (Automated Background Service)
1. On application startup, the `watchdog` observer begins monitoring `secure_drop_zone/`.
2. When new `.tif` files or `manifest.json` appear, the handler spawns a processing thread.
3. **Priority Queue**: If `/api/search` was called within the last 30 seconds, the watchdog pauses (analyst-first policy).
4. The scene is processed through the dual-model verification pipeline:
   - **RemoteCLIP**: Generates semantic embeddings to check for target identification.
   - **Prithvi-EO-2.0**: Runs temporal change comparison against baseline.
5. If both models flag a match, or if change exceeds 25%, an alert is broadcast via SSE.

### 5.3 Search & Retrieval
1. Analyst enters a query via the **Omni-Search bar** (natural language, coordinate array, or drawn bounding box).
2. Backend encodes the text query to a 512-dim vector via RemoteCLIP.
3. Qdrant performs cosine ANN search with optional filters: `scene_id`, `date_range`, `max_cloud`.
4. Results are post-filtered by bounding box intersection (client-side geo check).
5. Hits are rendered as georeferenced chip overlays on the Leaflet map.

### 5.4 Multi-Temporal Change Detection
1. Analyst selects two ingested scenes (T1 and T2) from the Change tab.
2. Raster engine reads co-registered tiles from both scenes in paired windows.
3. Prithvi-EO-2.0 extracts 6-band multispectral features for each tile pair.
4. Change score = `1 - cosine_similarity(T1_features, T2_features)`.
5. Results are ranked by change magnitude and displayed on the map with amber overlays.

### 5.5 Analyst Triage Workflow
1. Analyst reviews flagged chips in the Triage Modal or Metadata Drawer.
2. **Confirm Intel**: Verdict is recorded in the JSONL audit log; chip stays indexed.
3. **False Alarm**: Verdict is logged; the chip PNG is copied to `training_data/` for future model fine-tuning.
4. Verdicts are reflected on the map (green = confirmed, red = false alarm).

---

## 6. API Surface

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System status: GPU, VRAM, Qdrant reachability, point count |
| `POST` | `/api/search` | Semantic vector search with `bounding_box`, `date_range`, `scene_ids`, `max_cloud` filters |
| `POST` | `/api/ingest` | Tile a scene, embed chips with RemoteCLIP, upsert vectors into Qdrant |
| `GET` | `/api/scenes` | List all indexed scenes with tile counts and timestamps |
| `GET` | `/api/tiles/{id}` | Serve a cached PNG chip by tile ID |
| `POST` | `/api/temporal-change` | Run Prithvi change detection between two scenes |
| `POST` | `/api/temporal-change/export` | Export change results as downloadable GeoJSON |
| `POST` | `/api/triage` | Record an analyst verdict (Confirm Intel / False Alarm) |
| `POST` | `/api/triage/false_alarm` | Record false alarm + quarantine chip to `training_data/` |
| `GET` | `/api/triage` | Retrieve all analyst verdicts |
| `POST` | `/api/export` | Export triaged results as GeoJSON FeatureCollection |
| `GET` | `/api/watchdog/stream` | SSE endpoint for real-time watchdog alert notifications |

---

## 7. Project Directory Structure

```text
Trinetra-AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # All FastAPI route handlers
│   │   ├── core/
│   │   │   ├── config.py              # Central Settings (Pydantic, env-driven)
│   │   │   └── state.py               # PriorityQueueState + SSE EventBroadcaster
│   │   ├── schemas/
│   │   │   └── api.py                 # Request/Response Pydantic models
│   │   ├── services/
│   │   │   ├── raster_engine.py       # SceneReader, Tile dataclass, windowed I/O
│   │   │   ├── ml_inference.py        # RemoteCLIP + Prithvi-EO-2.0 (CUDA FP16)
│   │   │   ├── vector_store.py        # Qdrant client, ANN search, payload filters
│   │   │   ├── ingest.py              # Scene tiling + RemoteCLIP embedding + upsert
│   │   │   ├── change_detect.py       # Prithvi temporal change comparison
│   │   │   ├── audit.py               # JSONL verdict logging
│   │   │   └── watchdog_service.py    # File system monitor for secure_drop_zone/
│   │   └── main.py                    # FastAPI lifespan, CORS, static mount
│   ├── model_weights/                 # Pre-downloaded (read-only mount)
│   │   ├── RemoteCLIP-ViT-B-32.pt
│   │   └── Prithvi-EO-2.0-300M/
│   ├── sample_data/                   # Sentinel-2 GeoTIFF scenes
│   ├── secure_drop_zone/              # Watchdog-monitored ingestion directory
│   ├── training_data/                 # Quarantined false alarm chips
│   ├── tiles_cache/                   # Generated PNG chip previews
│   ├── ingest_pipeline.py             # STAC data acquisition (staging-only, online)
│   ├── requirements.txt               # Backend Python dependencies
│   ├── requirements-staging.txt        # Staging script dependencies (pystac_client)
│   ├── test_api.py                    # Phase 4 API integration tests
│   ├── test_env.py                    # Phase 1 environment verification
│   ├── test_inference.py              # Phase 3 model verification
│   └── test_raster_engine.py          # Phase 2 raster engine verification
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchPanel.jsx        # Omni-Search (text/coords/date/image upload)
│   │   │   ├── ChangePanel.jsx        # Temporal change comparison controls
│   │   │   ├── IngestPanel.jsx        # Manual scene ingestion form
│   │   │   ├── ResultCard.jsx         # Individual search result chip card
│   │   │   ├── MetadataDrawer.jsx     # Tile metadata + triage actions
│   │   │   └── TriageModal.jsx        # SSE-driven watchdog alert modal
│   │   ├── map/
│   │   │   ├── LeafletMap.jsx         # Map canvas + Geoman drawing tools
│   │   │   └── GraticuleLayer.js      # Offline coordinate grid overlay
│   │   ├── api/
│   │   │   └── client.js             # API client functions
│   │   ├── App.jsx                    # Root component, state management
│   │   ├── main.jsx                   # React entry point
│   │   └── styles.css                 # Global styles
│   └── package.json
├── docker-compose.yml                 # Container orchestration (trinetra + qdrant)
├── Dockerfile                         # Unified multi-stage build (frontend + backend)
├── qdrant_storage/                    # Persistent Qdrant volume (Docker named vol.)
├── TEAM_KNOWLEDGE_BASE.md             # SIH presentation reference document
├── Trinetra_Architecture_Specs.md     # This document
└── README.md                          # Setup, usage, and deployment guide
```

---

## 8. Concurrency & Resource Management

| Mechanism | Implementation | Purpose |
|-----------|---------------|---------|
| `_GPU_LOCK` | `threading.Lock()` in `routes.py` | Prevents concurrent GPU inference (OOM protection on 8 GB VRAM) |
| `PriorityQueueState` | `threading.Lock()` + timestamp in `state.py` | Watchdog defers to analyst — pauses 30s after any `/api/search` call |
| `EventBroadcaster` | `asyncio.Queue` per subscriber in `state.py` | Thread-safe SSE push from sync watchdog thread to async FastAPI |
| `GDAL_CACHEMAX` | 256 MB in `config.py` | Caps GDAL block cache to prevent RAM exhaustion during long ingests |
| Batch sizing | `clip_batch_size=32`, `ingest_upsert_batch=256` | Balances GPU throughput vs. VRAM peak; auto-falls back to 1 on OOM |

---

## 9. Security & Data Sovereignty

- **Zero egress**: No outbound network calls from the deployed Docker stack.
- **Sneakernet ingestion**: Data acquisition runs on a separate staging machine; imagery is physically transferred.
- **Audit trail**: Every analyst verdict is immutably appended to a JSONL log file.
- **False alarm quarantine**: Flagged chips are copied (not moved) to `training_data/` — originals remain indexed for audit continuity.
- **No telemetry**: No analytics, crash reporting, or usage tracking of any kind.