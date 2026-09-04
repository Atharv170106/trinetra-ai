# Trinetra AI

**Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery**

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH_2026-Problem_26227-blue.svg)](https://sih.gov.in)
[![Offline Capable](https://img.shields.io/badge/Air--Gapped-Supported-success.svg)](#)

Trinetra AI is an entirely offline, on-premises multimodal intelligence search engine for high-resolution satellite imagery, developed for the Ministry of Defence (MoD) / Indian Army (DGIS). It bridges the gap between conventional coordinate-based metadata filtering and semantic intent.

The system ingests massive Sentinel-2 / Landsat Cloud-Optimized GeoTIFFs (COGs), cuts them into spatial patches, extracts high-dimensional semantic embeddings using an offline vision-language model, and indexes them in a local vector database. This allows intelligence analysts to query ground locations using plain natural language (e.g., "aircraft on tarmac", "newly built military structures") via an interactive offline map dashboard.

---

## ✨ Features

- **Semantic Search**: Search for structures, vehicles, or geographical features using natural language queries.
- **Multi-Temporal Change Analysis**: Compare T1 and T2 satellite scenes to detect concrete developments, resource depletion, and infrastructure changes over time.
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
   **[http://localhost:8000](http://localhost:8000)**

---

## 📊 Generating and Ingesting Demo Data

Because Trinetra AI is fully offline, the Qdrant vector database starts empty. Follow these steps to generate synthetic Sentinel-2 demo scenes and ingest them to test the dashboard.

1. **Generate Demo Scenes**
   Use the included Python script to synthesize co-registered scenes. This is run locally using your Python virtual environment:
   ```powershell
   .\venv\Scripts\python.exe backend\make_demo_scenes.py
   ```
   *(This generates `DEMO_T43RGN_20240115T052131_L2A` and `DEMO_T43RGN_20260115T052131_L2A` inside the `backend/sample_data/` directory)*

2. **Ingest the Scenes via Dashboard**
   - Open the web dashboard ([http://localhost:8000](http://localhost:8000)) and go to the **Ingest** tab.
   - Enter `DEMO_T43RGN_20240115T052131_L2A` as the source and click **Ingest**.
   - Enter `DEMO_T43RGN_20260115T052131_L2A` as the source and click **Ingest**.

3. **Explore Features**
   - **Search Tab**: Type semantic queries (e.g., "water", "concrete pad") to search through the ingested chips.
   - **Change Tab**: Compare the T1 (2024) and T2 (2026) scenes to analyze temporal changes.

---

## 📂 Directory Structure

```text
Trinetra-AI/
├── backend/
│   ├── app/                   # FastAPI backend application
│   ├── model_weights/         # Pre-downloaded model weights (read-only)
│   ├── sample_data/           # Sentinel-2 GeoTIFF (.tif) scenes
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
