"""
Trinetra AI - Central configuration.

All tunables live here so Phase 2-6 modules never hardcode paths, band names,
or thresholds. Values can be overridden via a .env file or environment
variables prefixed with TRINETRA_ (e.g. TRINETRA_TILE_SIZE=512).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRINETRA_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # ----------------------------------------------------------- filesystem
    model_weights_dir: Path = BACKEND_ROOT / "model_weights"
    sample_data_dir: Path = BACKEND_ROOT / "sample_data"
    tiles_cache_dir: Path = BACKEND_ROOT / "tiles_cache"
    secure_drop_zone_dir: Path = BACKEND_ROOT / "secure_drop_zone"
    training_data_dir: Path = BACKEND_ROOT / "training_data"

    remoteclip_checkpoint: Path = model_weights_dir / "RemoteCLIP-ViT-B-32.pt"
    prithvi_dir: Path = model_weights_dir / "Prithvi-EO-2.0-300M"
    prithvi_checkpoint: Path = prithvi_dir / "Prithvi_EO_V2_300M.pt"

    # --------------------------------------------------------- inference / VRAM
    # Measured: RemoteCLIP FP16 = 0.30 GB resident on an 8 GB RTX 4060, leaving
    # ~6.3 GB of headroom, so both models stay resident (Tier 1).
    vram_safe_ceiling_gb: float = 7.2
    clip_batch_size: int = 32       # ~0.9 GB peak at 224px FP16; drops to 1 on OOM
    ingest_upsert_batch: int = 256   # Qdrant points per upsert call

    # ---------------------------------------------------------------- Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "satellite_tiles"
    qdrant_timeout: int = 30
    # Client 1.16.0 is pinned to match the v1.16.0 server image.
    qdrant_check_compatibility: bool = False

    # ---------------------------------------------------------- vector space
    embedding_dim: int = 512  # RemoteCLIP ViT-B-32

    # ------------------------------------------------------------- tiling
    # Chip size in SOURCE pixels at 10 m. Each chip is later resampled to the
    # input size each model expects (224 for both RemoteCLIP and Prithvi).
    tile_size: int = 256
    tile_overlap: int = 0            # stride = tile_size - tile_overlap
    remoteclip_input_size: int = 224
    prithvi_input_size: int = 224

    # -------------------------------------------------- quality / SCL masking
    # Sentinel-2 L2A Scene Classification Layer classes to treat as unusable.
    #   3 = cloud shadow, 8 = cloud medium prob, 9 = cloud high prob, 10 = thin cirrus
    scl_reject_classes: tuple[int, ...] = (3, 8, 9, 10)
    # Reject the tile when rejected classes exceed this fraction of valid pixels.
    max_cloud_fraction: float = 0.20
    # SCL class 0 = NO_DATA. Reject tiles that are mostly empty (scene edges).
    max_nodata_fraction: float = 0.20
    # Reject near-flat chips (water, uniform desert) - no semantic content.
    min_stddev: float = 1.0

    # ----------------------------------------------------------- watchdog ml gate
    watchdog_target_query: str = "military vehicles, bunkers, aircraft, or facilities"
    watchdog_similarity_threshold: float = 0.22
    watchdog_change_threshold: float = 0.40

    # ------------------------------------------------------------- band maps
    # RemoteCLIP is an RGB model: true-colour composite.
    remoteclip_bands: tuple[str, ...] = ("B04", "B03", "B02")

    # Prithvi-EO-2.0 band order is fixed by its config.json - do NOT reorder.
    # These are the HLS six the model was pretrained on: Blue, Green, Red,
    # Narrow-NIR, SWIR1, SWIR2. B02/B03/B04 are 10 m; B8A/B11/B12 are 20 m and
    # are upsampled to the 10 m reference grid by SceneReader.
    #
    # Previously listed B05/B06/B07 (red-edge), which the model never saw. The
    # mean/std below were always the correct HLS-six statistics - note the
    # index-3 spike (2734) then descent (1958 -> 1363), which is the NIR >>
    # SWIR1 > SWIR2 signature. Red-edge bands cannot produce that curve, so the
    # normalisation and the band list disagreed and the stats were right.
    # Changing this invalidates every existing embedding: re-ingest all scenes.
    prithvi_bands: tuple[str, ...] = ("B02", "B03", "B04", "B8A", "B11", "B12")
    prithvi_mean: tuple[float, ...] = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
    prithvi_std: tuple[float, ...] = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
    prithvi_num_frames: int = 4

    # Sentinel-2 L2A reflectance scaling (DN -> reflectance) for display stretch.
    s2_quantification_value: int = 10000

    # ------------------------------------------------------------------ GDAL
    # Keeps the per-process GDAL block cache small so 16 GB RAM is not eaten by
    # cached COG blocks during a long ingest.
    gdal_cachemax_mb: int = 256
    gdal_num_threads: str = "2"

    # -------------------------------------------------------------------- API
    api_title: str = "Trinetra AI"
    api_version: str = "0.6.0"
    # Dev only: the Vite server on :5173 is a different origin from :8000.
    # In Docker the built bundle is served by FastAPI itself, so requests are
    # same-origin and compose sets this to [] - no CORS exemptions at all.
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    search_limit_max: int = 200
    change_top_k_max: int = 500
    audit_log_path: Path = BACKEND_ROOT / "audit_log.jsonl"

    # Built React bundle. Present in the Docker image, absent during local dev
    # (where Vite serves it) - main.py mounts it only if it exists.
    frontend_dist_dir: Path = PROJECT_ROOT / "frontend" / "dist"

    # ------------------------------------------------------- XAI / Ollama SLM
    # Local vision-language model used by POST /api/explain. Air-gapped: the
    # weights are pre-seeded into the ollama_data volume by
    # `docker compose --profile setup up ollama-puller` and never fetched at
    # runtime.
    #
    # Inside compose the hostname is the service name. When running uvicorn from
    # the venv on the host instead, override with
    #   TRINETRA_OLLAMA_URL=http://localhost:11434
    ollama_url: str = "http://ollama:11434"
    # qwen2.5vl:3b - 3.2 GB download. The 7b tag is 6.0 GB and will not coexist
    # with Prithvi + RemoteCLIP on an 8 GB card; 32b/72b are not candidates.
    #
    # CAUTION: 3.2 GB is the download size, not the resident VRAM cost. Add the
    # KV cache (~300 MB at 8k ctx for a 3B GQA model) and the vision encoder's
    # activations on top. It only fits alongside our two models because compose
    # sets OLLAMA_KEEP_ALIVE=0 and /api/explain holds _GPU_LOCK, so this model is
    # never resident at the same moment as a Prithvi forward pass.
    ollama_model: str = "qwen2.5vl:3b"
    ollama_timeout_s: float = 300.0
    # Longest side of each chip sent to the SLM. Chips are 256 px; upscaling buys
    # nothing. Qwen2.5-VL tokenises at native resolution in 28 px cells, so 448
    # is ~256 visual tokens per image - two images stay well inside num_ctx.
    ollama_max_image_px: int = 448
    ollama_num_ctx: int = 8192

    # True  -> send T1 and T2 as two entries in the Ollama images[] array.
    # False -> composite them into one labelled side-by-side canvas instead.
    #
    # Native multi-image is the better answer on a model that supports it, but
    # /api/generate takes a raw prompt string with no per-image position markers,
    # so "which one is the baseline" depends on the model's chat template. The
    # composite path is deterministic and works on ANY vision model, including
    # single-image ones. Models listed in SINGLE_IMAGE_MODELS force it on.
    ollama_multi_image: bool = True

    # Models known to accept exactly one image. Anything here is composited
    # regardless of ollama_multi_image, because passing two silently drops the
    # second and the change explanation would describe only the baseline.
    single_image_models: tuple[str, ...] = ("moondream", "llava-phi3", "bakllava")

    # ------------------------------------------------------------------- misc
    tile_png_quality: int = 90
    log_level: str = "INFO"
    @property
    def tile_stride(self) -> int:
        stride = self.tile_size - self.tile_overlap
        if stride <= 0:
            raise ValueError("tile_overlap must be smaller than tile_size")
        return stride


settings = Settings()
