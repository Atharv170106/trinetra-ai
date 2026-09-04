# Trinetra AI - System Architecture & Project Specification Document
**Competition:** Smart India Hackathon (SIH) 2026 | **Problem Statement ID:** 26227
**Organization:** Ministry of Defence (MoD) / Indian Army (DGIS)
**Title:** Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
**Theme:** Space Technology | **Category:** Software

---

## 1. Executive Summary & Core Objective
Trinetra AI is an entirely offline, on-premises multimodal intelligence search engine for high-resolution satellite imagery. It bridges the gap between conventional coordinate-based metadata filtering and semantic intent. 

The system ingests massive Copernicus Sentinel-2 / Landsat Cloud-Optimized GeoTIFFs (COGs), cuts them into spatial patches, extracts high-dimensional semantic embeddings using an offline vision-language model, indexes them in a local vector database, and allows intelligence analysts to query ground locations using plain natural language (e.g., "aircraft on tarmac", "newly built military structures") on an interactive offline map dashboard.

---

## 2. Hardware Constraints & Deployment Rules
* **Host Machine:** RTX 4060 Laptop GPU (8GB VRAM), 16GB DDR5 System RAM, Intel Core i7-14700HX.
* **System RAM Limitation (16GB):** Full satellite scenes cannot be loaded into memory all at once. Streaming, chunked windowed reading, and aggressive memory management via `rasterio` are mandatory.
* **GPU VRAM Limitation (8GB):** Models must be sized or quantized to execute inference comfortably within 8GB VRAM alongside the OS overhead.
* **Air-Gap / 100% Offline Constraint:** The final application must execute strictly with network access disabled. No cloud APIs, external CDNs for map tiles, or external web services are permitted. All models, vector indexes, map tiles, and packages must execute locally in containerized Docker services.

---

## 3. Technology Stack
* **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic
* **Geospatial Processing:** GDAL, rasterio, shapely, pyproj, Pillow, NumPy
* **Local AI Model (Inference):** RemoteCLIP (ViT-B-32 checkpoint) loaded via `open_clip_torch` on CUDA
* **Vector Database:** Qdrant (running locally via Docker container, persistent volume on disk)
* **Frontend:** React (Vite), JavaScript, Leaflet.js (with offline cached raster map tiles), Axios
* **Infrastructure & Containerization:** Docker & Docker Compose

---

## 4. Required Project Directory Structure
```text
Trinetra-AI/
├── backend/
│   ├── app/
│   │   ├── api/                   # FastAPI routes (search.py, health.py)
│   │   ├── core/                  # Configurations, settings, logging
│   │   ├── schemas/               # Request/Response Pydantic models
│   │   ├── services/
│   │   │   ├── raster_engine.py   # Tiling, windowed reading, coordinate projection
│   │   │   ├── ml_inference.py    # RemoteCLIP text/image embedding generator (CUDA)
│   │   │   └── vector_store.py    # Qdrant client connection, indexing, similarity search
│   │   └── main.py                # FastAPI app entry point
│   ├── model_weights/             # Local pre-downloaded weights (RemoteCLIP-ViT-B-32.pt)
│   ├── sample_data/               # Sample Sentinel-2 GeoTIFF (.tif) scenes
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/            # SearchBar, ResultCard, MetadataDrawer
│   │   ├── map/                   # LeafletMap, BoundingBoxLayer
│   │   ├── api/                   # Axios client for FastAPI backend
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── Dockerfile
├── qdrant_storage/                # Persistent Qdrant vector volume
└── docker-compose.yml             # Container orchestration