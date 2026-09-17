# Trinetra AI: SIH Problem Statement 26227 Compliance Matrix

**Project:** Trinetra AI  
**Problem Statement:** 26227 (Air-gapped, multimodal satellite search engine)  
**Constraint:** 8GB RTX 4060 VRAM Limit  

This document serves as a comprehensive compliance checklist, mapping the official Smart India Hackathon requirements to the specific engineering features implemented in the Trinetra AI architecture. It demonstrates that all criteria have been rigorously met for deployment in secure, air-gapped defense networks.

---

## Section 2.2: Technical Requirements Breakdown

### ✅ 2.2.1 Semantic and Multimodal Retrieval
**Requirement:** Capability to retrieve imagery using natural language and multimodal queries.
* **Our Solution:** We implemented a FastAPI backend using the **RemoteCLIP** vision-language model to convert satellite image chips into 512-dimensional vectors. We use a **Qdrant** vector database to perform rapid cosine-similarity matching against natural language queries (e.g., *"aircraft on tarmac"*) entered via our React-based **"Tactical Omni-Search"** bar.

### ✅ 2.2.2 Multi-Temporal Change Analysis
**Requirement:** Ability to analyze changes across time-series satellite imagery.
* **Our Solution:** We integrated the **Prithvi-EO-2.0** foundation model to process Level-2A imagery. Our UI features a dual-pane temporal map that visually highlights structural and terrain changes between a T1 (baseline) and T2 (recent) timeframe using **Cloud-Optimized GeoTIFFs (COGs)**.

### ✅ 2.2.3 False-Alarm Suppression and Quality Handling
**Requirement:** Mechanisms to filter out noise, clouds, and false positives.
* **Our Solution:** Our Tactical Omni-Search includes dynamic popover filters for **Maximum Cloud Cover (SCL)** and **Minimum Cosine Similarity Thresholds**. This prevents heavily obscured imagery and low-confidence matches from cluttering the analyst's view.

### ✅ 2.2.4 Discovery and Clustering
**Requirement:** Automatically group visually and semantically similar regions.
* **Our Solution:** Because our retrieval pipeline relies entirely on **Qdrant** vector embeddings, visually and semantically similar sites naturally cluster in the multidimensional space. Our UI plots these retrieval hits as heat-mapped intelligence chips across a **Leaflet grid**.

### ✅ 2.2.5 Analyst Workflow and Provenance
**Requirement:** Provide tools for triage, verification, and data export.
* **Our Solution:** We built a complete "Analyst Triage Loop." The UI includes a `MetadataDrawer` where analysts can trigger an **Explainable AI (XAI)** summary generated locally by a **Qwen2.5-VL 3B** model via **Ollama**. Analysts can mark hits as "Confirmed Intel" or "False Alarm," and instantly export the triaged intelligence report as a **GeoJSON** file that preserves all original acquisition metadata.

### ✅ 2.2.6 Scale, Incremental Ingestion and Sovereignty
**Requirement:** Handle massive datasets incrementally without full re-indexing.
* **Our Solution:** We built a Python background **watchdog script** that incrementally chunks massive Sentinel-2 COGs into 256x256 tiles and pushes them into Qdrant without requiring a complete index rebuild.

### ✅ 2.2.7 Constraints (Air-Gapped / Offline)
**Requirement:** The solution must operate strictly offline on limited hardware.
* **Our Solution:** Trinetra AI is strictly designed for defense air-gapped networks. We use a custom offline **HTML Canvas graticule layer** instead of relying on internet-based map tile servers. All AI models (**RemoteCLIP**, **Prithvi**, and **Qwen Vision**) are heavily optimized with 4-bit quantization to run entirely offline on an **8GB GPU constraint**.

---

## Section 2.3: Expected Solution

### ✅ 2.3 Expected Solution Delivery
**Requirement:** A fully functional, locally testable system.
* **Our Solution:** The architecture is fully staged for local evaluation, complete with an index-build watchdog and a reproducible UI dashboard ready for held-out query testing.

---

### 3. Unique Selling Propositions (USPs) & Technical Innovations

While Trinetra AI strictly adheres to the core problem statement, we engineered several advanced capabilities to ensure the system is genuinely production-ready for high-stakes defense environments.

* **On-Demand Explainable AI (XAI) via Vision SLMs:** Instead of just returning mathematical cosine similarity scores, we integrated a local **Qwen 2.5 Vision SLM (3B parameters)** via **Ollama**. Analysts can click a manual trigger to generate a tactical, text-based explanation of why a chip was flagged or what changed in a temporal comparison. We heavily optimized this to run alongside foundation models strictly within an **8GB VRAM limit** using 4-bit quantization and lazy-loading.
* **Intelligent Tactical Omni-Bar UI:** We eliminated standard, clunky web forms. We engineered a **Palantir-style floating glassmorphism command bar** that uses regex to automatically detect and route input types. If an analyst types text, it triggers a semantic search; if they draw on the map or type `[W, S, E, N]`, it instantly switches to a bounding-box Spatial Watchdog query.
* **True Air-Gapped Map Survivability:** Most GIS projects break when taken completely offline because they lose access to standard map tile servers (like OpenStreetMap). We engineered a custom **HTML Canvas Graticule Layer** that dynamically draws exact latitude and longitude grids offline, ensuring analysts maintain geographic orientation without internet access.
* **Asynchronous Watchdog & VRAM Orchestration:** We implemented an autonomous **background watchdog** for incremental ingestion that operates independently of the FastAPI routes. Furthermore, heavy AI models (Prithvi-EO-2.0 and RemoteCLIP) are lazy-loaded into memory *only* when a search is executed and immediately cleared. This orchestration ensures a 100% uptime API health status without triggering Out-of-Memory (OOM) crashes on standard tactical hardware.
