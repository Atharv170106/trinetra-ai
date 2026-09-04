"""
Trinetra AI - Phase 3b: Qdrant vector store.

Collection `satellite_tiles`: 512-dim RemoteCLIP vectors, Cosine distance.
Prithvi features are NOT indexed - temporal change is computed on demand from
two dates, so there is nothing to persist (see ml_inference.PrithviChangeEncoder).

Client is pinned to 1.16.0 to match the server image; check_compatibility is
disabled because the client warns on any minor-version drift.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Payload keys that get a Qdrant index. Without these, filtered search degrades
# to a full scan once the collection grows past a few thousand points.
_INDEXED_FIELDS: dict[str, str] = {
    "scene_id": "keyword",
    "acquisition_timestamp": "keyword",
    "scl_cloud_coverage": "float",
}


class VectorStoreError(RuntimeError):
    """Qdrant connectivity or schema failure."""


@dataclass(slots=True)
class SearchHit:
    tile_id: str
    score: float
    scene_id: str | None
    acquisition_timestamp: str | None
    wgs84_bounding_box: list[float] | None
    scl_cloud_coverage: float | None
    payload: dict[str, Any]

    @classmethod
    def from_point(cls, point: Any) -> "SearchHit":
        payload = dict(point.payload or {})
        return cls(
            tile_id=payload.get("tile_id", str(point.id)),
            score=float(point.score),
            scene_id=payload.get("scene_id"),
            acquisition_timestamp=payload.get("acquisition_timestamp"),
            wgs84_bounding_box=payload.get("wgs84_bounding_box"),
            scl_cloud_coverage=payload.get("scl_cloud_coverage"),
            payload=payload,
        )


def tile_point_id(tile_id: str) -> str:
    """
    Deterministic UUID5 from the tile_id string.

    Qdrant point IDs must be uint64 or UUID, but tile_ids are strings like
    'SCENE_r0012c0034'. UUID5 keeps re-ingest idempotent: the same tile always
    overwrites its own point instead of duplicating.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"trinetra/tile/{tile_id}"))


class VectorStore:
    """Thin wrapper over QdrantClient with lazy connection and schema bootstrap."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self._client = None

    # ------------------------------------------------------------- connection
    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise VectorStoreError(f"qdrant-client unavailable: {exc}") from exc
            self._client = QdrantClient(
                url=self.url,
                timeout=settings.qdrant_timeout,
                check_compatibility=settings.qdrant_check_compatibility,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Qdrant close raised: %s", exc)
            self._client = None

    def health(self) -> dict:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            exists = self.collection in collections
            count = self.count() if exists else 0
            return {
                "reachable": True,
                "url": self.url,
                "collection": self.collection,
                "collection_exists": exists,
                "points": count,
            }
        except Exception as exc:
            return {"reachable": False, "url": self.url, "error": str(exc)}

    # ----------------------------------------------------------------- schema
    def ensure_collection(self, *, recreate: bool = False) -> bool:
        """
        Create the collection and payload indexes if absent. Idempotent.
        Returns True when a collection was created.

        recreate=True DROPS all vectors - only for a deliberate re-index.
        """
        from qdrant_client import models

        try:
            exists = self.client.collection_exists(self.collection)

            if exists and recreate:
                logger.warning("Dropping collection %s (recreate=True)", self.collection)
                self.client.delete_collection(self.collection)
                exists = False

            if exists:
                info = self.client.get_collection(self.collection)
                actual = info.config.params.vectors
                dim = getattr(actual, "size", None)
                if dim is not None and dim != settings.embedding_dim:
                    raise VectorStoreError(
                        f"Collection {self.collection} has dim {dim}, expected "
                        f"{settings.embedding_dim}. Re-index with recreate=True."
                    )
                self._ensure_indexes()
                return False

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                    # Keep vectors on disk: 16 GB host, and the OS page cache
                    # handles the hot set better than Qdrant's RAM copy.
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                ),
            )
            logger.info("Created collection %s (%d-dim, Cosine, on_disk)",
                        self.collection, settings.embedding_dim)
            self._ensure_indexes()
            return True
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"ensure_collection failed: {exc}") from exc

    def _ensure_indexes(self) -> None:
        from qdrant_client import models

        schema_map = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "float": models.PayloadSchemaType.FLOAT,
        }
        for field, kind in _INDEXED_FIELDS.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema_map[kind],
                    wait=True,
                )
            except Exception as exc:
                # Already-exists is the common case and is not an error.
                logger.debug("payload index %s: %s", field, exc)

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception as exc:
            raise VectorStoreError(f"count failed: {exc}") from exc

    # ----------------------------------------------------------------- writes
    def upsert_tiles(
        self,
        records: Iterable[tuple[str, np.ndarray, dict]],
        *,
        batch_size: int | None = None,
    ) -> int:
        """
        Upsert (tile_id, vector, payload) triples. Streams in batches so a
        whole-scene ingest never materializes every point at once.
        """
        from qdrant_client import models

        batch = batch_size or settings.ingest_upsert_batch
        buffer: list[Any] = []
        written = 0

        def flush() -> None:
            nonlocal buffer, written
            if not buffer:
                return
            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=buffer,
                    wait=True,
                )
                written += len(buffer)
            except Exception as exc:
                raise VectorStoreError(
                    f"upsert of {len(buffer)} point(s) failed: {exc}"
                ) from exc
            finally:
                buffer = []

        for tile_id, vector, payload in records:
            vec = np.asarray(vector, dtype=np.float32).ravel()
            if vec.size != settings.embedding_dim:
                raise VectorStoreError(
                    f"{tile_id}: vector has {vec.size} dims, expected "
                    f"{settings.embedding_dim}"
                )
            if not np.isfinite(vec).all():
                raise VectorStoreError(f"{tile_id}: vector contains NaN/Inf")

            buffer.append(
                models.PointStruct(
                    id=tile_point_id(tile_id),
                    vector=vec.tolist(),
                    payload={**payload, "tile_id": tile_id},
                )
            )
            if len(buffer) >= batch:
                flush()
        flush()

        logger.info("Upserted %d point(s) into %s", written, self.collection)
        return written

    def delete_scene(self, scene_id: str) -> None:
        """Remove every tile belonging to one scene - used for re-ingest."""
        from qdrant_client import models

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="scene_id",
                                match=models.MatchValue(value=scene_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info("Deleted tiles for scene %s", scene_id)
        except Exception as exc:
            raise VectorStoreError(f"delete_scene({scene_id}) failed: {exc}") from exc

    # ----------------------------------------------------------------- reads
    @staticmethod
    def _build_filter(
        scene_ids: Sequence[str] | None = None,
        max_cloud: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ):
        """Compose an optional Qdrant filter. Returns None when unconstrained."""
        from qdrant_client import models

        must: list[Any] = []
        if scene_ids:
            must.append(
                models.FieldCondition(
                    key="scene_id",
                    match=models.MatchAny(any=list(scene_ids)),
                )
            )
        if max_cloud is not None:
            must.append(
                models.FieldCondition(
                    key="scl_cloud_coverage",
                    range=models.Range(lte=float(max_cloud)),
                )
            )
        if bbox is not None:
            # wgs84_bounding_box is stored as [W, S, E, N]; a geo filter would
            # need a separate geo payload field, so this is left to the caller
            # to post-filter. Flagged rather than silently ignored.
            logger.debug("bbox filtering is applied client-side, not in Qdrant")
        return models.Filter(must=must) if must else None

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 20,
        score_threshold: float | None = None,
        scene_ids: Sequence[str] | None = None,
        max_cloud: float | None = None,
    ) -> list[SearchHit]:
        """
        Cosine similarity search. Uses query_points(); client.search() was
        removed in qdrant-client 1.16.
        """
        vec = np.asarray(query_vector, dtype=np.float32).ravel()
        if vec.size != settings.embedding_dim:
            raise VectorStoreError(
                f"query vector has {vec.size} dims, expected {settings.embedding_dim}"
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vec.tolist(),
                query_filter=self._build_filter(scene_ids, max_cloud),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"query_points failed: {exc}") from exc

        # Results live on .points, not on the response object itself.
        return [SearchHit.from_point(p) for p in response.points]

    def get_tile(self, tile_id: str) -> dict | None:
        """Fetch one tile's payload by its logical tile_id."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[tile_point_id(tile_id)],
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"retrieve({tile_id}) failed: {exc}") from exc
        return dict(points[0].payload or {}) if points else None

    def list_scenes(self) -> list[dict]:
        """
        Aggregate indexed scenes: tile count, timestamp, and union bbox.

        Scrolls payloads only (with_vectors=False) so this stays cheap; at a few
        thousand tiles per scene the cost is a handful of paged reads.
        """
        scenes: dict[str, dict] = {}
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=512,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for p in points:
                    payload = p.payload or {}
                    sid = payload.get("scene_id")
                    if not sid:
                        continue
                    entry = scenes.setdefault(sid, {
                        "scene_id": sid,
                        "tile_count": 0,
                        "acquisition_timestamp": payload.get("acquisition_timestamp"),
                        "wgs84_bounding_box": None,
                    })
                    entry["tile_count"] += 1
                    bbox = payload.get("wgs84_bounding_box")
                    if bbox and len(bbox) == 4:
                        cur = entry["wgs84_bounding_box"]
                        entry["wgs84_bounding_box"] = bbox if cur is None else [
                            min(cur[0], bbox[0]), min(cur[1], bbox[1]),
                            max(cur[2], bbox[2]), max(cur[3], bbox[3]),
                        ]
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError(f"list_scenes failed: {exc}") from exc
        return sorted(scenes.values(), key=lambda s: s["scene_id"])

    def iter_scene_tiles(self, scene_id: str, *, page: int = 256):
        """Scroll every tile payload for a scene without loading them all at once."""
        from qdrant_client import models

        offset = None
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="scene_id", match=models.MatchValue(value=scene_id)
                )
            ]
        )
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=page,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                yield dict(p.payload or {})
            if offset is None:
                break


# --------------------------------------------------------------- module singleton
vector_store = VectorStore()
