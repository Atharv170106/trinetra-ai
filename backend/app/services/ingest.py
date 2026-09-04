"""
Trinetra AI - Phase 3c: Scene ingest pipeline.

Streams raster_engine -> RemoteCLIP -> Qdrant with a bounded working set: at
most `clip_batch_size` chips are held in RAM at any moment, regardless of scene
size. A 10980x10980 Sentinel-2 tile yields ~1800 chips and never exceeds a few
hundred MB of RSS.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.core.config import settings
from app.services.ml_inference import InferenceError, remoteclip
from app.services.raster_engine import SceneReader, to_rgb_uint8
from app.services.vector_store import VectorStoreError, vector_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestReport:
    scene_id: str
    tiles_seen: int = 0
    tiles_rejected: int = 0
    tiles_indexed: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "tiles_seen": self.tiles_seen,
            "tiles_rejected": self.tiles_rejected,
            "tiles_indexed": self.tiles_indexed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "tiles_per_second": (
                round(self.tiles_indexed / self.elapsed_seconds, 2)
                if self.elapsed_seconds > 0 else 0.0
            ),
            "errors": self.errors,
        }


def ingest_scene(
    source: str,
    *,
    scene_id: str | None = None,
    max_tiles: int | None = None,
    replace: bool = True,
    save_previews: bool = True,
) -> IngestReport:
    """
    Tile a scene, embed each surviving chip with RemoteCLIP, upsert into Qdrant.

    replace=True drops any existing points for this scene first, so re-ingesting
    a scene never leaves orphaned tiles from a previous run with different
    tiling settings.
    save_previews writes a PNG per indexed tile to tiles_cache for the frontend.
    """
    started = time.perf_counter()

    with SceneReader(source, scene_id=scene_id) as scene:
        report = IngestReport(scene_id=scene.scene_id)
        vector_store.ensure_collection()

        if replace:
            try:
                vector_store.delete_scene(scene.scene_id)
            except VectorStoreError as exc:
                report.errors.append(f"pre-delete failed: {exc}")

        preview_dir = settings.tiles_cache_dir / scene.scene_id
        if save_previews:
            preview_dir.mkdir(parents=True, exist_ok=True)

        batch_imgs: list[np.ndarray] = []
        batch_meta: list[tuple[str, dict]] = []
        pending: list[tuple[str, np.ndarray, dict]] = []

        def flush_batch() -> None:
            """Embed the buffered chips and hand them to the upsert buffer."""
            nonlocal batch_imgs, batch_meta
            if not batch_imgs:
                return
            try:
                vectors = remoteclip.encode_images(batch_imgs)
            except InferenceError as exc:
                report.errors.append(f"embed batch of {len(batch_imgs)}: {exc}")
                batch_imgs, batch_meta = [], []
                return

            if vectors.shape[0] != len(batch_meta):
                report.errors.append(
                    f"embedding count {vectors.shape[0]} != chips {len(batch_meta)}"
                )
            for i, (tid, payload) in enumerate(batch_meta):
                if i < vectors.shape[0]:
                    pending.append((tid, vectors[i], payload))
            batch_imgs, batch_meta = [], []
            del vectors

        def flush_upsert() -> None:
            nonlocal pending
            if not pending:
                return
            try:
                report.tiles_indexed += vector_store.upsert_tiles(pending)
            except VectorStoreError as exc:
                report.errors.append(f"upsert: {exc}")
            finally:
                pending = []
                gc.collect()

        for tile in scene.iter_tiles(load_pixels=True, max_tiles=max_tiles):
            report.tiles_seen += 1
            if tile.rgb is None:
                report.tiles_rejected += 1
                continue

            try:
                rgb8 = to_rgb_uint8(tile.rgb)
            except Exception as exc:
                report.errors.append(f"{tile.tile_id}: stretch failed: {exc}")
                continue

            if save_previews:
                try:
                    from PIL import Image

                    Image.fromarray(rgb8).save(
                        preview_dir / f"{tile.tile_id}.png",
                        optimize=True,
                    )
                except Exception as exc:
                    report.errors.append(f"{tile.tile_id}: preview failed: {exc}")

            batch_imgs.append(rgb8)
            batch_meta.append((tile.tile_id, tile.payload()))

            if len(batch_imgs) >= settings.clip_batch_size:
                flush_batch()
            if len(pending) >= settings.ingest_upsert_batch:
                flush_upsert()

        flush_batch()
        flush_upsert()

    report.elapsed_seconds = time.perf_counter() - started
    logger.info("Ingest %s: %d indexed / %d seen in %.1fs",
                report.scene_id, report.tiles_indexed, report.tiles_seen,
                report.elapsed_seconds)
    return report
