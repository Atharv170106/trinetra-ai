# 🧠 Trinetra AI: Team Knowledge Base (SIH Deep Dive)

Welcome to the Trinetra AI project! This document is our team's central source of truth. It explains exactly what we have built, how it works under the hood, and our vision for the future. Read this carefully, understand the concepts deeply, and use the bolded keywords to speak confidently to the Smart India Hackathon (SIH) judges.

---

## 1. The Mission & Problem Statement

**Trinetra AI** is an **air-gapped, sovereign multimodal satellite search engine** built specifically for defense and intelligence analysts.

* **What does that mean?**
  Instead of analysts manually scanning hundreds of gigabytes of satellite images or using complex geographic coordinates, they can simply type in plain English (e.g., "aircraft on tarmac") and our system instantly finds the relevant locations.
* **Why it matters (The Problem Statement):**
  Military intelligence demands extreme data privacy and speed. Most modern AI runs in the cloud (like ChatGPT), which is a massive security risk for top-secret defense data. Our system complies with strict **data sovereignty** requirements by running **100% offline (air-gapped)** on local, consumer-grade hardware. No data ever leaves the laptop.

---

## 2. The Tech Stack (Explained Simply)

* **Backend: FastAPI, PyTorch, & Transformers**
  **FastAPI** is the high-performance engine handling our web requests. We use **PyTorch** and **Transformers** to load and run the heavy AI models directly on the GPU.
* **Database: Qdrant (Local Vector Database)**
  Traditional databases (like SQL) search for exact text matches. A **vector database** searches by meaning (semantics). It stores data as high-dimensional mathematical coordinates called **embeddings**. 
  *Note: We run Qdrant using **Docker named volumes** instead of Windows bind mounts because Windows file systems struggle with the specific memory-mapped file operations (mmap) that Qdrant uses for high-speed retrieval, which causes corruption.*
* **Frontend: React & Leaflet Map**
  We use **React** to build a snappy, interactive dashboard and **Leaflet** to render offline map tiles, allowing analysts to visualize satellite data globally without an internet connection.
* **AI Models:**
  * **RemoteCLIP**: Used for **semantic text search** (translating English words and images into the same mathematical space to find matches).
  * **Prithvi-EO-2.0**: A specialized Earth Observation model used for **temporal change analysis** (comparing an area at Time A vs. Time B to detect infrastructure changes).

---

## 3. The Core Workflow (How Data Moves)

Here is the step-by-step journey of a satellite image through our system:

1. **Ingestion & Windowed COG Tiling**: 
   We ingest massive raw satellite files called **Cloud-Optimized GeoTIFFs (COGs)**. Because these files are too large for RAM, we use a technique called **windowed reading** to slice the massive image into small, manageable 256x256 pixel spatial patches (chips) on the fly, without crashing the computer.
2. **Generating Vector Embeddings**: 
   Each small patch is passed through our **RemoteCLIP vision-language model**. The model analyzes the pixels and outputs a 512-dimensional array of numbers—a **vector embedding**—that mathematically represents the "meaning" of that patch.
3. **Indexing in Qdrant**: 
   These vectors, along with the patch's geographic bounding box, are saved into our **Qdrant vector database**.
4. **Analyst Query**: 
   An analyst types a query into the UI (e.g., "new runway"). RemoteCLIP converts that text into a text vector. Qdrant performs a **cosine similarity search** to find the image vectors that mathematically align closest with the text vector, instantly returning the relevant map locations.

---

## 4. Current MVP Features (What We Have Built Now)

* **The Dual-Model VRAM Strategy**:
  We are running two heavy models on a standard 8GB RTX 4060 laptop. We achieve this via **On-Demand execution** (models are lazy-loaded into memory only when needed), running models in **Native FP16 (half-precision floating point)** to halve the memory footprint, and utilizing a safety pivot to **bitsandbytes 8-bit quantization** which mathematically shrinks the model further if VRAM gets too tight.
* **False-Alarm Suppression (SCL Masking)**:
  Clouds look like structural changes to AI models, causing false alarms. We wrote an efficient filter that reads the satellite's **Scene Classification Layer (SCL)**. It automatically masks out pixels classified as cloud shadows (pixel 3), medium/high probability clouds (pixels 8, 9), and cirrus clouds (pixel 10), ensuring the AI only analyzes clear ground.
* **Analyst Triage UI**:
  Our UI features **[✓ Confirm]** and **[✗ False Alarm]** triage buttons. Analysts review AI predictions and can instantly export confirmed intelligence reports as standard **GeoJSON** files, ready to be handed to command structures.

---

## 5. The Data Evolution: Multispectral vs. Hyperspectral

To win SIH, we must explain exactly how our data pipeline evolves from the current MVP to our ultimate vision.

### What we use NOW: Multispectral Data (Sentinel-2)
Currently, Trinetra AI ingests **Multispectral** data (like Copernicus Sentinel-2). 
* **What it is:** Multispectral sensors capture around **10 to 13 broad color bands** (e.g., Red, Green, Blue, Near-Infrared).
* **The Limitation:** While excellent for detecting structural changes (like a new building) or broad vegetation indices, multispectral data lacks deep chemical context. **It cannot distinguish between green paint (camouflage) and real green leaves**, because both look broadly "green" in those few bands.

### What we will use LATER: Hyperspectral Data (EnMAP, Pixxel)
Our roadmap aggressively targets **Hyperspectral** data.
* **What it is:** Hyperspectral sensors capture **over 200 narrow, continuous spectral bands**. 
* **The Advantage:** Instead of just a color, every single pixel has a deep, continuous **spectral signature** (a chemical fingerprint). This allows us to see exactly what materials are on the ground. We can instantly detect military camouflage netting, specific chemical plumes, or stealth coatings.

---

## 6. The Grand Finale Roadmap & USP: The Spectral Adapter

This is our ultimate **Unique Selling Proposition (USP)** for the judges. 

### The Bottleneck: Why Current Models Fail on Hyperspectral Data
Our current foundation models (RemoteCLIP and Prithvi-EO) are **hard-coded** to accept a specific number of input channels (e.g., exactly 6 multispectral bands). 
If you try to feed a 200-band hyperspectral image into Prithvi, **the model's input layers will crash due to a dimensional mismatch**. 
To fix this normally, you would have to rebuild and **retrain the entire massive foundation model from scratch**—which requires millions of dollars in supercomputing compute.

### The Solution: The Spectral Adapter Architecture
Instead of retraining the massive models, we will use **Parameter-Efficient Fine-Tuning (PEFT)** by introducing a **Spectral Adapter**.
1. **The Translator:** We will bolt a tiny, lightweight neural network layer (the adapter) directly in front of RemoteCLIP and Prithvi.
2. **Dimensionality Reduction:** This adapter acts as a translator. It takes the 200 incoming hyperspectral bands and mathematically compresses/projects them down into the exact 6-channel format the foundation models expect.
3. **Training the Adapter:** We **freeze** the massive foundation model so its weights never change. We *only* train the tiny Spectral Adapter. The adapter learns how to preserve the critical material signatures (like camouflage) while formatting it perfectly for the AI to understand.
4. **The Impact:** We achieve **Sub-Pixel Material Fingerprinting** using advanced hyperspectral data, on a standard laptop, without paying the millions required to train a foundation model.

---

## 7. Judge Q&A Cheat Sheet

Memorize these answers. Use the **bolded technical keywords** when speaking to the judges!

**Q1: How do you handle hardware constraints? Satellites are massive and AI requires huge GPUs.**
> *"We use **windowed reading** of **Cloud-Optimized GeoTIFFs** so we never load full scenes into RAM. On the GPU side, we implement **On-Demand execution** and **Native FP16 precision**, with an automatic fallback to **8-bit quantization** using bitsandbytes, allowing us to comfortably fit two heavy foundation models into an 8GB laptop."*

**Q2: How do you prevent false positives from clouds when detecting changes?**
> *"We implement **SCL Masking**. We programmatically parse the Sentinel-2 **Scene Classification Layer** and apply a bitmask to aggressively filter out cloud shadows and high-probability clouds before the pixels ever reach our neural network."*

**Q3: What type of satellite data are you using, and what is your future plan?**
> *"Right now, we use **Multispectral** data from Sentinel-2, which has about 13 broad bands—great for structural changes but easy to fool with camouflage. Our roadmap moves to **Hyperspectral** data, which has over 200 continuous bands. This gives every pixel a chemical fingerprint, allowing us to perform **sub-pixel material identification**."*

**Q4: Your foundation models (Prithvi/RemoteCLIP) can't handle 200 hyperspectral bands. How will you fix this without spending millions to retrain them?**
> *"You're absolutely right, feeding 200 bands into a 6-band model causes a dimensional crash. Instead of retraining the foundation model, we **freeze** it and inject a **Spectral Adapter layer** at the input. We only train this tiny adapter using **Parameter-Efficient Fine-Tuning (PEFT)**. The adapter learns to compress the 200 hyperspectral bands into the exact 6-channel format the model expects, preserving the critical material signatures while saving massive compute costs."*

**Q5: Why use Qdrant instead of a standard SQL database?**
> *"Standard SQL handles exact keyword matches. We need to capture human intent. Qdrant is a **vector database** that performs **cosine similarity searches** on high-dimensional **embeddings** generated by our vision-language model, allowing analysts to search by meaning rather than metadata."*
