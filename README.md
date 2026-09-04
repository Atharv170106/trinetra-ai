# Trinetra AI

Trinetra AI is an offline-capable, air-gapped system for semantic search and temporal change detection on satellite imagery.

## Prerequisites

- **Docker Desktop**: With WSL2 backend (for Windows) or native Docker (Linux). Ensure NVIDIA Container Toolkit is installed for GPU passthrough.
- **Python 3.11+**: For running local demo generation and tests.

## How to Run the Application

1. **Start the containers**
   The backend API, frontend bundle, and the Qdrant vector database are orchestrated via Docker Compose.
   ```bash
   docker compose up -d
   ```

2. **Access the Preview**
   Once the containers are running, open your web browser and navigate to:
   [http://localhost:8000](http://localhost:8000)

## Generating and Ingesting Demo Data

The Qdrant vector database starts empty. Follow these steps to generate and ingest demo data to test the dashboard.

1. **Generate Demo Scenes**
   Use the existing virtual environment to run the demo scene generator:
   ```powershell
   .\venv\Scripts\python.exe backend\make_demo_scenes.py
   ```
   *(This script generates `DEMO_T43RGN_20240115T052131_L2A` and `DEMO_T43RGN_20260115T052131_L2A` inside `backend/sample_data/`)*

2. **Ingest the Scenes**
   - Open the web dashboard and go to the **Ingest** tab.
   - Enter `DEMO_T43RGN_20240115T052131_L2A` as the source and click ingest.
   - Enter `DEMO_T43RGN_20260115T052131_L2A` as the source and click ingest.

3. **View the Results**
   - Head over to the **Change** tab in the dashboard.
   - Compare the T1 (2024) and T2 (2026) scenes to see the temporal change detection in action (e.g., identifying a new concrete pad and a shrunken reservoir).
