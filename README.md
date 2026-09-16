# Trinetra AI

**Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery**

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH_2026-Problem_26227-blue.svg)](https://sih.gov.in)
[![Offline Capable](https://img.shields.io/badge/Air--Gapped-Supported-success.svg)](#)

Trinetra AI is an entirely offline, on-premises multimodal intelligence search engine for high-resolution satellite imagery, developed for the Ministry of Defence (MoD) / Indian Army (DGIS). It bridges the gap between conventional coordinate-based metadata filtering and semantic intent.

The system ingests massive Sentinel-2 / Landsat Cloud-Optimized GeoTIFFs (COGs), cuts them into spatial patches, extracts high-dimensional semantic embeddings using an offline vision-language model, and indexes them in a local vector database. This allows intelligence analysts to query ground locations using plain natural language (e.g., "aircraft on tarmac", "newly built military structures") via an interactive offline map dashboard.

---

## ✨ Features

- **Semantic Search**: Search for structures, vehicles, or geographical features using natural language queries.
- **Omni-Search & Map Drawing**: Use regex-powered text/coordinate parsing or draw spatial bounding boxes directly on the map via Leaflet-Geoman to constrain searches.
- **Air-Gapped AOI Watchdog**: A background service automatically monitors a secure drop zone for new GeoTIFFs, running them through a two-step Prithvi and RemoteCLIP verification to stream Real-Time Alerts (SSE) to the UI.
- **Multi-Temporal Change Analysis**: Compare T1 and T2 satellite scenes to detect concrete developments, resource depletion, and infrastructure changes over time.
- **False Alarm Triage**: Analysts can flag false positive alerts, which automatically quarantines the image chips to a local training data repository.
- **100% Air-Gapped**: Fully functional offline without external APIs, CDNs, or cloud dependencies. 
- **Efficient Processing**: Streams and processes large GeoTIFFs in chunks using `rasterio` and GDAL to comfortably run on local consumer-grade hardware.
- **Unified Web Interface**: An interactive React-based map dashboard to seamlessly ingest, search, and triage changes.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic
- **Geospatial Processing**: GDAL, rasterio, shapely, pyproj, Pillow, NumPy
- **AI Models (Inference)**: RemoteCLIP (ViT-B-32 checkpoint), Prithvi-EO-2.0-300M (running via `open_clip_torch` on CUDA)
- **Vector Database**: Qdrant (local Docker container)
- **Frontend**: React (Vite), JavaScript, Leaflet.js
- **Infrastructure**: Docker & Docker Compose

---

## 💻 Hardware Constraints & Deployment

Designed to run on a consumer-grade laptop with the following baseline:
- **Host GPU**: RTX 4060 Laptop GPU (8GB VRAM) or equivalent.
- **System RAM**: 16GB DDR5.
- **Storage**: SSD highly recommended for Qdrant vector volume and local weights.

> [!IMPORTANT]  
> All models, vector indexes, map tiles, and packages execute locally. To comply with the air-gap posture, model weights (approx. 1.5GB) are pre-downloaded and mounted read-only into the Docker container.

---

## ⚙️ Prerequisites

- **Docker Desktop**: With WSL2 backend (for Windows) or native Docker (Linux). 
  - Ensure the **NVIDIA Container Toolkit** is installed and configured for Docker to support GPU passthrough.
- **Python 3.11+**: For running local demo generation scripts outside of Docker.

---

## 🚀 Installation & Running

The backend API, frontend bundle, and the Qdrant vector database are orchestrated into a single, seamless environment via Docker Compose.

1. **Clone the repository**
   ```bash
   git clone https://github.com/Atharv170106/trinetra-ai.git
   cd trinetra-ai
   ```

2. **Start the containers**
   ```bash
   docker compose up -d
   ```
   *Note: On the first run, Docker will need network access to build the images (pulling python packages and node modules). Subsequent executions can be run fully air-gapped.*

3. **Access the Application**
   Once the containers are healthy, open your web browser and navigate to:
   **[http://localhost:8080](http://localhost:8080)**

---

## 📡 Ingesting Satellite Imagery

Trinetra AI uses a background watchdog service to automatically monitor and ingest new scenes from a secure drop zone. To acquire real data, use the included STAC ingestion script on an internet-connected machine.

1. **Download Sentinel-2 COGs**
   Run the `ingest_pipeline.py` script to fetch real Sentinel-2 scenes for a bounding box:
   ```powershell
   .\venv\Scripts\python.exe backend\ingest_pipeline.py --bbox 72.80 18.90 73.05 19.15 --start 2025-01-01 --end 2025-03-31 --limit 2
   ```
   *(This downloads the required bands directly into `backend/secure_drop_zone/`)*

2. **Automated Background Ingestion (Watchdog)**
   The air-gapped system automatically detects new folders in `backend/secure_drop_zone/`. 
   - New scenes are tiled and processed.
   - If significant changes or matches are found, **Real-Time Alerts** will appear in the UI.

3. **Explore Features**
   - **Search Tab**: Type semantic queries (e.g., "aircraft on runway") or use the Leaflet-Geoman draw tools to select a bounding box constraint.
   - **Change Tab**: Select two ingested scenes to analyze temporal changes.
   - **Triage**: Click "False Alarm" on any alerts to move the false positive image chips into `backend/training_data/`.

---

## 📂 Directory Structure

```text
Trinetra-AI/
├── backend/
│   ├── app/                   # FastAPI backend application
│   ├── model_weights/         # Pre-downloaded model weights (read-only)
│   ├── sample_data/           # Sentinel-2 GeoTIFF (.tif) scenes
│   ├── secure_drop_zone/      # Automated ingestion directory for the Watchdog
│   ├── training_data/         # Triaged false alarm image chips
│   ├── make_demo_scenes.py    # Demo data generator
│   ├── requirements.txt       
│   └── test_api.py            # Local API testing script
├── frontend/
│   ├── src/                   # React Vite source code
│   └── package.json           
├── docker-compose.yml         # Container orchestration
└── Dockerfile                 # Unified Dockerfile for Frontend & Backend
```

## 📜 License

[MIT License](LICENSE)
