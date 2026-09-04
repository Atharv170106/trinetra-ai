"""
Trinetra AI - Phase 1 Environment Verification
Run from the repo root:  python backend/test_env.py

Verifies, in order:
  1. Python version
  2. CUDA / RTX 4060 availability + a real fp16 matmul on-device
  3. rasterio + bundled GDAL
  4. open_clip import and offline tokenizer (must make zero network calls)
  5. Presence of RemoteCLIP weights (warn only - not required until Phase 3)
  6. Qdrant readiness: connect -> create -> upsert -> query_points -> drop

Exit code 0 = all critical checks passed.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path

QDRANT_URL = "http://localhost:6333"
PROBE_COLLECTION = "_trinetra_env_probe"
VECTOR_SIZE = 512  # RemoteCLIP ViT-B-32

_WEIGHTS_DIR = Path(__file__).parent / "model_weights"
WEIGHTS_PATH = _WEIGHTS_DIR / "RemoteCLIP-ViT-B-32.pt"
PRITHVI_DIR = _WEIGHTS_DIR / "Prithvi-EO-2.0-300M"
PRITHVI_CKPT = PRITHVI_DIR / "Prithvi_EO_V2_300M.pt"
PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"

VRAM_SAFE_CEILING_GB = 7.2   # Tier-2 pivot threshold from the master spec
PRITHVI_FP16_EST_GB = 0.6    # ~300M params at fp16, weights only

PASS, FAIL, WARN = "[ PASS ]", "[ FAIL ]", "[ WARN ]"
_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def bad(check: str, msg: str) -> None:
    print(f"{FAIL} {check}: {msg}")
    _failures.append(check)


def warn(msg: str) -> None:
    print(f"{WARN} {msg}")
    _warnings.append(msg)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------- 1. Python
def check_python() -> None:
    section("Python")
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        ok(label)
    else:
        # Not fatal, but rasterio/torch wheel pins in requirements.txt target 3.11.
        warn(f"{label} - project pins target 3.11; wheel resolution may differ.")


# ------------------------------------------------------------------ 2. CUDA
def check_cuda() -> None:
    section("CUDA / GPU")
    try:
        import torch
    except ImportError as exc:
        bad("torch import", f"{exc}. Install the cu121 build first (see requirements.txt header).")
        return

    print(f"        torch {torch.__version__} | compiled for CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        bad("cuda.is_available", "no CUDA device visible. CPU-only wheel installed, or driver issue.")
        return

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    ok(f"{name} | sm_{major}{minor} | {total_gb:.2f} GB VRAM")

    if total_gb < 7.0:
        warn(f"Only {total_gb:.2f} GB VRAM detected - Phase 3 batch sizes assume ~8 GB.")

    # Real work on-device: fp16 matmul, then confirm memory is reclaimed.
    try:
        with torch.inference_mode():
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            b = a @ a
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            assert b.shape == (2048, 2048)
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_gb, _ = (x / 1024**3 for x in torch.cuda.mem_get_info())
        ok(f"fp16 matmul on device (peak {peak_mb:.0f} MB) | {free_gb:.2f} GB free after cleanup")
    except Exception as exc:
        bad("cuda compute", f"{type(exc).__name__}: {exc}")


# -------------------------------------------------------------- 3. rasterio
def check_rasterio() -> None:
    section("Geospatial")
    try:
        import rasterio
    except ImportError as exc:
        bad("rasterio import", str(exc))
        return

    ok(f"rasterio {rasterio.__version__} | bundled GDAL {rasterio.__gdal_version__}")

    try:
        import pyproj
        import shapely

        ok(f"shapely {shapely.__version__} | pyproj {pyproj.__version__} "
           f"(PROJ {pyproj.proj_version_str})")
    except ImportError as exc:
        bad("shapely/pyproj import", str(exc))


# ------------------------------------------------------------- 4. open_clip
def check_open_clip() -> None:
    section("open_clip (offline)")
    try:
        import open_clip
    except ImportError as exc:
        bad("open_clip import", str(exc))
        return

    ok(f"open_clip_torch {getattr(open_clip, '__version__', 'unknown')}")

    # Tokenizer must resolve from the bundled BPE vocab with no network access.
    try:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(["aircraft on tarmac"])
        ok(f"offline tokenizer OK (no network) | token tensor {tuple(tokens.shape)}")
    except Exception as exc:
        bad("open_clip tokenizer", f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------- 5. model weights
def check_weights() -> None:
    section("Model weights")
    if WEIGHTS_PATH.is_file():
        size_mb = WEIGHTS_PATH.stat().st_size / 1024**2
        ok(f"{WEIGHTS_PATH.name} present ({size_mb:.1f} MB)")
    else:
        warn(f"{WEIGHTS_PATH} missing. Download it WHILE STILL ONLINE - "
             "required from Phase 3 onward, unobtainable after air-gapping.")

    # Prithvi ships a raw torch checkpoint (.pt) + prithvi_mae.py, NOT a
    # transformers-style safetensors repo. There is no auto_map in config.json,
    # so AutoModel.from_pretrained(trust_remote_code=True) does NOT apply.
    if PRITHVI_CKPT.is_file():
        size_mb = PRITHVI_CKPT.stat().st_size / 1024**2
        ok(f"{PRITHVI_CKPT.name} present ({size_mb:.1f} MB)")
        for required in ("config.json", "prithvi_mae.py"):
            if (PRITHVI_DIR / required).is_file():
                ok(f"  {required} present")
            else:
                warn(f"  {required} MISSING - required to instantiate the 3D ViT offline.")
    else:
        warn(f"{PRITHVI_CKPT} missing. Snapshot '{PRITHVI_REPO}' WHILE STILL ONLINE "
             "(needs Prithvi_EO_V2_300M.pt + config.json + prithvi_mae.py).")


# ------------------------------------------------- 5b. bitsandbytes (Tier-2)
def check_bitsandbytes() -> None:
    section("bitsandbytes (Tier-2 fallback)")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        warn(f"bitsandbytes unavailable ({exc}). Tier-2 8-bit fallback disabled; "
             "Tier-1 FP16 still usable.")
        return
    except Exception as exc:
        warn(f"bitsandbytes import raised {type(exc).__name__}: {exc} "
             "(usually a CUDA binary mismatch on Windows).")
        return

    ok(f"bitsandbytes {getattr(bnb, '__version__', 'unknown')}")

    # Prove the 8-bit linear kernel actually runs; import success alone is not enough.
    try:
        import torch

        if not torch.cuda.is_available():
            warn("skipping 8-bit kernel test: no CUDA device.")
            return
        layer = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False).cuda()
        with torch.inference_mode():
            out = layer(torch.randn(2, 64, device="cuda", dtype=torch.float16))
        assert out.shape == (2, 64)
        del layer, out
        torch.cuda.empty_cache()
        ok("Linear8bitLt CUDA kernel executes")
    except Exception as exc:
        warn(f"8-bit kernel test failed ({type(exc).__name__}: {exc}). "
             "Tier-2 unreliable - plan on CPU offload instead.")


# ---------------------------------------- 5bb. transformers (Prithvi loader)
def check_transformers() -> None:
    section("transformers / accelerate (Prithvi loader path)")
    try:
        import transformers
    except ImportError as exc:
        bad("transformers import", str(exc))
        return

    ok(f"transformers {transformers.__version__}")

    try:
        import accelerate

        ok(f"accelerate {accelerate.__version__} (required for load_in_8bit)")
    except ImportError as exc:
        warn(f"accelerate missing ({exc}) - load_in_8bit=True will fail.")

    try:
        import einops

        ok(f"einops {einops.__version__} (Prithvi remote code dep)")
    except ImportError as exc:
        warn(f"einops missing ({exc}) - Prithvi's custom 3D ViT code needs it.")


# ------------------------------------------- 5c. dual-model VRAM feasibility
def check_dual_model_vram() -> None:
    """Tier-1 budget check: both models resident in FP16 must fit under VRAM_SAFE_CEILING_GB."""
    section("Dual-model VRAM budget (Tier-1 FP16)")
    try:
        import torch
    except ImportError:
        bad("dual-model check", "torch unavailable.")
        return

    if not torch.cuda.is_available():
        bad("dual-model check", "no CUDA device.")
        return

    if not WEIGHTS_PATH.is_file():
        warn("skipping: RemoteCLIP weights absent.")
        return

    try:
        import open_clip
    except ImportError:
        warn("skipping: open_clip unavailable.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=None, device="cpu"
        )
        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # strict=False silently tolerates a wrong checkpoint - inspect what it skipped.
        result = model.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if "positional_ids" not in k]
        if missing:
            bad("RemoteCLIP weights",
                f"{len(missing)} missing key(s) after load, e.g. {missing[:3]}. "
                "Checkpoint likely does not match ViT-B-32.")
            return
        if result.unexpected_keys:
            warn(f"{len(result.unexpected_keys)} unexpected key(s) ignored "
                 f"(e.g. {list(result.unexpected_keys)[:3]}) - normal for RemoteCLIP.")

        model = model.to("cuda", dtype=torch.float16).eval()
        clip_gb = torch.cuda.memory_allocated() / 1024**3
        ok(f"RemoteCLIP loaded FP16 on CUDA | {clip_gb:.2f} GB resident")

        # Real forward pass at batch_size=1, matching the Tier-1 protocol.
        with torch.inference_mode():
            feats = model.encode_image(
                torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)
            )
            tokens = open_clip.get_tokenizer("ViT-B-32")(["aircraft on tarmac"]).to("cuda")
            txt = model.encode_text(tokens)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        ok(f"encode_image {tuple(feats.shape)} | encode_text {tuple(txt.shape)} "
           f"| peak {peak_gb:.2f} GB")

        if feats.shape[-1] != VECTOR_SIZE or txt.shape[-1] != VECTOR_SIZE:
            bad("embedding dim",
                f"expected {VECTOR_SIZE}, got image={feats.shape[-1]} text={txt.shape[-1]}.")

        # Headroom projection: Prithvi 300M in FP16 is ~0.6 GB of weights.
        projected = peak_gb + PRITHVI_FP16_EST_GB
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"        projected Tier-1 dual-model floor: {projected:.2f} GB "
              f"(ceiling {VRAM_SAFE_CEILING_GB} GB / device {total_gb:.2f} GB)")
        if projected > VRAM_SAFE_CEILING_GB:
            warn("Projected dual-model residency exceeds the safe ceiling - "
                 "lazy-load Prithvi per request and/or use Tier-2 8-bit.")
        else:
            ok(f"headroom for Prithvi activations: "
               f"{VRAM_SAFE_CEILING_GB - projected:.2f} GB")
    except torch.cuda.OutOfMemoryError as exc:
        bad("dual-model VRAM", f"OOM during Tier-1 load: {exc}")
    except Exception as exc:
        bad("dual-model VRAM", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------ 6. Qdrant
def check_qdrant(retries: int = 10, delay: float = 2.0) -> None:
    section("Qdrant")
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        bad("qdrant_client import", str(exc))
        return

    try:
        from importlib.metadata import version as _pkg_version

        print(f"        qdrant-client {_pkg_version('qdrant-client')}")
    except Exception:
        print("        qdrant-client version: unknown")
    if not hasattr(QdrantClient, "query_points"):
        bad("qdrant-client version",
            "query_points() absent - upgrade to >=1.16.0 (search() is removed).")
        return

    # check_compatibility=False: client 1.19 vs server 1.16 triggers a noisy warning.
    # The APIs we use (collection_exists/create_collection/upsert/query_points) are
    # stable across that gap, but the mismatch is pinned out in requirements anyway.
    client = QdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)

    # Readiness poll: the container accepts TCP before the storage layer is up.
    for attempt in range(1, retries + 1):
        try:
            client.get_collections()
            ok(f"reachable at {QDRANT_URL} (attempt {attempt})")
            break
        except Exception as exc:
            if attempt == retries:
                bad("qdrant connection",
                    f"unreachable after {retries} attempts ({type(exc).__name__}). "
                    "Is `docker compose up -d` running?")
                return
            print(f"        waiting for Qdrant... ({attempt}/{retries})")
            time.sleep(delay)

    # Full round trip on a throwaway collection.
    try:
        if client.collection_exists(PROBE_COLLECTION):
            client.delete_collection(PROBE_COLLECTION)

        client.create_collection(
            collection_name=PROBE_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        ok(f"create_collection('{PROBE_COLLECTION}', {VECTOR_SIZE}-dim, COSINE)")

        client.upsert(
            collection_name=PROBE_COLLECTION,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.05] * VECTOR_SIZE,
                    payload={"tile_id": "probe_0001", "bbox": [77.0, 28.0, 77.1, 28.1]},
                )
            ],
            wait=True,
        )
        ok("upsert(wait=True)")

        response = client.query_points(
            collection_name=PROBE_COLLECTION,
            query=[0.05] * VECTOR_SIZE,
            limit=1,
            with_payload=True,
        )
        hits = response.points  # NOTE: results live on .points, not the response itself
        if not hits:
            bad("query_points", "returned zero hits for an exact-match vector.")
        else:
            hit = hits[0]
            ok(f"query_points -> score={hit.score:.4f} payload={hit.payload}")
    except Exception as exc:
        bad("qdrant round trip", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            if client.collection_exists(PROBE_COLLECTION):
                client.delete_collection(PROBE_COLLECTION)
                ok("probe collection cleaned up")
        except Exception as exc:
            warn(f"could not drop probe collection: {exc}")
        client.close()


# -------------------------------------------------------------------- runner
def main() -> int:
    print("=" * 68)
    print(" TRINETRA AI - PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 68)

    for check in (check_python, check_cuda, check_rasterio,
                  check_open_clip, check_weights, check_bitsandbytes,
                  check_transformers, check_dual_model_vram, check_qdrant):
        try:
            check()
        except Exception as exc:  # a check itself blowing up is a failure, not a crash
            bad(check.__name__, f"unhandled {type(exc).__name__}: {exc}")
            traceback.print_exc()

    section("Summary")
    if _failures:
        print(f"{FAIL} {len(_failures)} critical check(s) failed: {', '.join(_failures)}")
    else:
        print(f"{PASS} All critical checks passed.")
    if _warnings:
        print(f"{WARN} {len(_warnings)} warning(s) - non-blocking for Phase 1.")
    print("=" * 68)

    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
