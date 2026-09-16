"""
Trinetra AI - Phase 3a: Local model inference (RemoteCLIP + Prithvi-EO-2.0).

VRAM protocol (measured on RTX 4060 8GB: RemoteCLIP FP16 = 0.30 GB resident):
  Tier 1  - both models FP16 on CUDA, batch_size configurable but small.
  Tier 2  - on OOM or >VRAM_SAFE_CEILING_GB, retry with batch_size=1; if that
            still fails, evict Prithvi to CPU. Prithvi cannot use
            bitsandbytes load_in_8bit because it is NOT a transformers model
            (raw .pt checkpoint, no auto_map), so 8-bit would require manual
            Linear8bitLt surgery. Given 6.3 GB of headroom we do not need it.

Both loaders are lazy singletons: nothing touches the GPU until first use, so
importing this module stays cheap for API workers that only serve tiles.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)

# Prithvi's encoder width for the 300M variant (config.json: embed_dim 1024).
PRITHVI_EMBED_DIM = 1024


class InferenceError(RuntimeError):
    """Model load or forward-pass failure that callers should surface as 5xx."""


def _vram_gb() -> tuple[float, float]:
    """(allocated, total) in GB. Zeros when CUDA is absent."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )


def _free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA unavailable - running on CPU (slow, demo only).")
    return torch.device("cpu")


# ============================================================== RemoteCLIP
class RemoteCLIPEncoder:
    """
    RemoteCLIP ViT-B-32 for text<->image retrieval. Emits L2-normalized
    512-dim vectors so Qdrant cosine distance is a plain dot product.
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: torch.device | None = None
        self._dtype = torch.float16
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            ckpt = settings.remoteclip_checkpoint
            if not ckpt.is_file():
                raise InferenceError(
                    f"RemoteCLIP checkpoint missing: {ckpt}. Download it while online."
                )
            try:
                import open_clip
            except ImportError as exc:
                raise InferenceError(f"open_clip unavailable: {exc}") from exc

            device = _resolve_device()
            # Half precision is not worth the accuracy loss on CPU fallback.
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            try:
                # pretrained=None: never reaches the network.
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained=None, device="cpu"
                )
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    state = state.get("state_dict", state)

                result = model.load_state_dict(state, strict=False)
                # strict=False silently accepts a mismatched checkpoint; a real
                # weight tensor going missing means the wrong file was supplied.
                missing = [k for k in result.missing_keys if "positional_ids" not in k]
                if missing:
                    raise InferenceError(
                        f"{ckpt.name} does not match ViT-B-32: {len(missing)} missing "
                        f"key(s), e.g. {missing[:3]}"
                    )
                if result.unexpected_keys:
                    logger.debug("Ignored %d unexpected key(s) in %s",
                                 len(result.unexpected_keys), ckpt.name)
                del state

                model = model.to(device=device, dtype=dtype).eval()
                model.requires_grad_(False)

                self._model = model
                self._preprocess = preprocess
                self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._device = device
                self._dtype = dtype
            except InferenceError:
                self.unload()
                raise
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                raise InferenceError(f"OOM loading RemoteCLIP: {exc}") from exc
            except Exception as exc:
                self.unload()
                raise InferenceError(f"RemoteCLIP load failed: {exc}") from exc

            used, total = _vram_gb()
            logger.info("RemoteCLIP ready on %s (%s) | VRAM %.2f/%.2f GB",
                        device, dtype, used, total)

    def unload(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        _free_vram()

    # -------------------------------------------------------------- encoding
    @staticmethod
    def _l2(vecs: torch.Tensor) -> torch.Tensor:
        return vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_text(self, queries: str | Sequence[str]) -> np.ndarray:
        """(N, 512) float32, L2-normalized."""
        self.load()
        texts = [queries] if isinstance(queries, str) else list(queries)
        if not texts:
            raise ValueError("no query text supplied")

        try:
            tokens = self._tokenizer(texts).to(self._device)
            with torch.inference_mode():
                feats = self._model.encode_text(tokens)
                feats = self._l2(feats.float())
            return feats.cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:
            _free_vram()
            raise InferenceError(f"OOM encoding {len(texts)} text query(ies)") from exc

    def encode_images(
        self,
        images: Iterable[np.ndarray],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Encode HWC uint8 RGB chips -> (N, 512) float32, L2-normalized.

        Chips arrive at 256px from the raster engine; open_clip's preprocess
        transform handles the resize to 224 and the CLIP normalization.
        """
        self.load()
        from PIL import Image

        batch = batch_size or settings.clip_batch_size
        outputs: list[np.ndarray] = []
        pending: list[torch.Tensor] = []

        def flush() -> None:
            nonlocal pending
            if not pending:
                return
            stacked = torch.stack(pending).to(self._device, dtype=self._dtype)
            pending = []
            try:
                with torch.inference_mode():
                    feats = self._model.encode_image(stacked)
                    outputs.append(self._l2(feats.float()).cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                # Tier-2 step 1: fall back to one image at a time.
                logger.warning("OOM on batch of %d - retrying singly.", stacked.shape[0])
                _free_vram()
                for i in range(stacked.shape[0]):
                    with torch.inference_mode():
                        feats = self._model.encode_image(stacked[i : i + 1])
                        outputs.append(self._l2(feats.float()).cpu().numpy())
            finally:
                del stacked
                _free_vram()

        for arr in images:
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"expected (H, W, 3) uint8 RGB, got {arr.shape}")
            if arr.dtype != np.uint8:
                raise ValueError(f"expected uint8, got {arr.dtype}")
            pending.append(self._preprocess(Image.fromarray(arr)))
            if len(pending) >= batch:
                flush()
        flush()

        if not outputs:
            return np.empty((0, settings.embedding_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


# ================================================================== Prithvi
@dataclass(slots=True)
class ChangeResult:
    """Per-chip temporal change metrics between two acquisitions."""

    cosine_distance: float      # 1 - cos(T1, T2); 0 = identical
    l2_distance: float
    change_score: float         # cosine_distance clamped to [0, 1]


class PrithviChangeEncoder:
    """
    Prithvi-EO-2.0-300M encoder for multi-temporal change.

    Loaded by direct PrithviMAE instantiation - it ships a raw .pt checkpoint
    with no auto_map, so transformers' AutoModel path does not apply.

    Configured with num_frames=2 (T1, T2) rather than the checkpoint's default
    4, which is legal because positional embeddings are recomputed per
    num_frames and the pos_embed keys are dropped from the state dict.
    """

    def __init__(self, num_frames: int = 2) -> None:
        self._model = None
        self._device: torch.device | None = None
        self._dtype = torch.float16
        self._num_frames = num_frames
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> torch.device | None:
        return self._device

    # ------------------------------------------------------------- lifecycle
    def load(self, *, force_cpu: bool = False) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            ckpt = settings.prithvi_checkpoint
            code = settings.prithvi_dir / "prithvi_mae.py"
            if not ckpt.is_file():
                raise InferenceError(f"Prithvi checkpoint missing: {ckpt}")
            if not code.is_file():
                raise InferenceError(f"prithvi_mae.py missing: {code}")

            # The checkpoint's model code lives beside the weights, not on the
            # import path; add it once so `from prithvi_mae import PrithviMAE` works.
            weights_dir = str(settings.prithvi_dir)
            if weights_dir not in sys.path:
                sys.path.insert(0, weights_dir)
            try:
                from prithvi_mae import PrithviMAE  # type: ignore[import-not-found]
            except ImportError as exc:
                raise InferenceError(
                    f"Cannot import PrithviMAE from {code}: {exc}. "
                    "einops is required."
                ) from exc

            device = _resolve_device(prefer_cuda=not force_cpu)
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            try:
                model = PrithviMAE(
                    img_size=settings.prithvi_input_size,
                    patch_size=(1, 16, 16),
                    num_frames=self._num_frames,
                    in_chans=len(settings.prithvi_bands),
                    embed_dim=PRITHVI_EMBED_DIM,
                    depth=24,
                    num_heads=16,
                    decoder_embed_dim=512,
                    decoder_depth=8,
                    decoder_num_heads=16,
                    mlp_ratio=4.0,
                    coords_encoding=[],
                    mask_ratio=0.0,   # inference: never mask
                )

                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                # pos_embed is a fixed sinusoidal buffer sized for num_frames=4;
                # the model recomputes it, so a stale copy would break shapes.
                dropped = [k for k in state if "pos_embed" in k]
                for k in dropped:
                    del state[k]

                result = model.load_state_dict(state, strict=False)
                real_missing = [
                    k for k in result.missing_keys if "pos_embed" not in k
                ]
                if real_missing:
                    raise InferenceError(
                        f"{ckpt.name} mismatch: {len(real_missing)} missing key(s), "
                        f"e.g. {real_missing[:3]}"
                    )
                logger.debug("Prithvi: dropped %d pos_embed key(s), %d unexpected",
                             len(dropped), len(result.unexpected_keys))
                del state

                # Only the encoder is needed for change detection; the MAE decoder
                # is ~90 MB of dead weight during inference.
                encoder = model.encoder.to(device=device, dtype=dtype).eval()
                encoder.requires_grad_(False)
                del model

                self._model = encoder
                self._device = device
                self._dtype = dtype
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                if not force_cpu:
                    logger.warning("OOM loading Prithvi on CUDA - retrying on CPU.")
                    self.load(force_cpu=True)
                    return
                raise InferenceError(f"OOM loading Prithvi: {exc}") from exc
            except InferenceError:
                self.unload()
                raise
            except Exception as exc:
                self.unload()
                raise InferenceError(f"Prithvi load failed: {exc}") from exc

            used, total = _vram_gb()
            logger.info("Prithvi encoder ready on %s (%s, %d frames) | VRAM %.2f/%.2f GB",
                        device, dtype, self._num_frames, used, total)

    def unload(self) -> None:
        self._model = None
        self._device = None
        _free_vram()

    # -------------------------------------------------------------- encoding
    def _encode_pair(self, cube: torch.Tensor) -> torch.Tensor:
        """
        cube: (1, C, T, H, W) normalized float. Returns (T, embed_dim) - one
        mean-pooled descriptor per timestep, excluding the CLS token.
        """
        with torch.inference_mode():
            features = self._model.forward_features(cube)
            # forward_features returns one tensor per transformer block; the last
            # is the normalized output.
            tokens = features[-1]           # (1, 1 + T*tokens_per_frame, embed_dim)
            patch_tokens = tokens[:, 1:, :]  # drop CLS
            del features

            n_tokens = patch_tokens.shape[1]
            per_frame = n_tokens // self._num_frames
            if per_frame * self._num_frames != n_tokens:
                raise InferenceError(
                    f"token count {n_tokens} not divisible by num_frames "
                    f"{self._num_frames}"
                )
            # (1, T, per_frame, E) -> mean over spatial tokens -> (T, E)
            grouped = patch_tokens.reshape(1, self._num_frames, per_frame, -1)
            pooled = grouped.mean(dim=2).squeeze(0).float()
            del patch_tokens, grouped
            return pooled

    def compare(self, t1: np.ndarray, t2: np.ndarray) -> ChangeResult:
        """
        Compute change between two co-registered chips.

        t1/t2: (6, H, W) float32 already standardized by
        raster_engine.normalize_for_prithvi(). Band order must be
        B02,B03,B04,B8A,B11,B12.
        """
        self.load()
        expected_c = len(settings.prithvi_bands)
        for name, arr in (("t1", t1), ("t2", t2)):
            if arr.ndim != 3 or arr.shape[0] != expected_c:
                raise ValueError(f"{name}: expected ({expected_c}, H, W), got {arr.shape}")
        if t1.shape != t2.shape:
            raise ValueError(f"shape mismatch: t1={t1.shape} t2={t2.shape}")

        size = settings.prithvi_input_size
        cube = None
        try:
            # (C, T, H, W) with T=2, then batch dim.
            stacked = np.stack([t1, t2], axis=1).astype(np.float32, copy=False)
            cube = torch.from_numpy(stacked).unsqueeze(0)
            del stacked

            if cube.shape[-1] != size or cube.shape[-2] != size:
                # Interpolate spatially; collapse T into batch so 4-D interpolate
                # applies, then restore.
                c, t = cube.shape[1], cube.shape[2]
                flat = cube.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
                flat = torch.nn.functional.interpolate(
                    flat, size=(size, size), mode="bilinear", align_corners=False
                )
                cube = flat.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
                del flat
                assert cube.shape[1] == c and cube.shape[2] == t

            cube = cube.to(device=self._device, dtype=self._dtype)

            try:
                pooled = self._encode_pair(cube)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM in Prithvi forward - evicting to CPU and retrying.")
                _free_vram()
                self.unload()
                self.load(force_cpu=True)
                cube = cube.to(device=self._device, dtype=self._dtype)
                pooled = self._encode_pair(cube)

            f1, f2 = pooled[0], pooled[1]
            cos = torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
            l2 = torch.norm(f1 - f2).item()
            cosine_distance = 1.0 - cos
            del pooled, f1, f2

            return ChangeResult(
                cosine_distance=round(cosine_distance, 6),
                l2_distance=round(l2, 6),
                change_score=round(min(max(cosine_distance, 0.0), 1.0), 6),
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise InferenceError(f"OOM during change comparison: {exc}") from exc
        finally:
            del cube
            _free_vram()
    def _encode_batch(self, cube: torch.Tensor) -> torch.Tensor:
        """
        cube: (B, C, T, H, W) normalized float. 
        Returns (B, T, embed_dim)
        """
        with torch.inference_mode():
            features = self._model.forward_features(cube)
            tokens = features[-1]
            patch_tokens = tokens[:, 1:, :]  # drop CLS
            del features

            n_tokens = patch_tokens.shape[1]
            per_frame = n_tokens // self._num_frames
            
            # (B, T, per_frame, E) -> mean over spatial tokens -> (B, T, E)
            grouped = patch_tokens.reshape(cube.shape[0], self._num_frames, per_frame, -1)
            pooled = grouped.mean(dim=2).float()
            del patch_tokens, grouped
            return pooled

    def compare_batch(self, t1_list: list[np.ndarray], t2_list: list[np.ndarray]) -> list[ChangeResult]:
        """
        Computes batched change scores.
        """
        self.load()
        cubes = []
        for t1, t2 in zip(t1_list, t2_list):
            stacked = np.stack([t1, t2], axis=1).astype(np.float32, copy=False)
            cubes.append(torch.from_numpy(stacked))
            
        # (B, C, T, H, W)
        batch_cube = torch.stack(cubes).to(device=self._device, dtype=self._dtype)
        
        pooled = self._encode_batch(batch_cube) # (B, T, E)
        
        results = []
        for i in range(pooled.shape[0]):
            f1, f2 = pooled[i, 0], pooled[i, 1]
            cos = torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
            l2 = torch.norm(f1 - f2).item()
            cosine_distance = 1.0 - cos
            results.append(ChangeResult(
                cosine_distance=round(cosine_distance, 6),
                l2_distance=round(l2, 6),
                change_score=round(min(max(cosine_distance, 0.0), 1.0), 6),
            ))
            
        del batch_cube, pooled
        _free_vram()
        return results


# ------------------------------------------------------------- module singletons
remoteclip = RemoteCLIPEncoder()
prithvi = PrithviChangeEncoder()


def vram_report() -> dict:
    used, total = _vram_gb()
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_allocated_gb": round(used, 3),
        "vram_total_gb": round(total, 3),
        "vram_ceiling_gb": settings.vram_safe_ceiling_gb,
        "over_ceiling": used > settings.vram_safe_ceiling_gb,
        "remoteclip_loaded": remoteclip.is_loaded,
        "prithvi_loaded": prithvi.is_loaded,
        "prithvi_device": str(prithvi.device) if prithvi.device else None,
    }
