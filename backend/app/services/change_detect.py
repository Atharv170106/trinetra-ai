"""
Trinetra AI - Phase 4: Multi-temporal change analysis.

Nothing is persisted. Two scenes are opened side by side, the same grid cells
are read from both, and Prithvi scores each pair on demand. Only one chip pair
is resident at a time, so a full-scene diff costs the same RAM as a single tile.

Grid compatibility is checked up front: comparing r,c across two scenes only
means anything if both share a CRS, size, and geotransform.
"""

from __future__ import annotations

import time
import math
import gc
import torch
import logging
from dataclasses import dataclass, field

from app.core.config import settings
from app.services.ml_inference import InferenceError, prithvi
from app.services.raster_engine import (
    RasterEngineError,
    SceneReader,
    TileQuality,
    normalize_for_prithvi,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChangeTile:
    tile_id: str
    row: int
    col: int
    wgs84_bounding_box: list[float]
    change_score: float
    cosine_distance: float
    l2_distance: float
    t1_cloud: float
    t2_cloud: float

    def as_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "wgs84_bounding_box": self.wgs84_bounding_box,
            "change_score": self.change_score,
            "cosine_distance": self.cosine_distance,
            "l2_distance": self.l2_distance,
            "t1_cloud": self.t1_cloud,
            "t2_cloud": self.t2_cloud,
        }


@dataclass(slots=True)
class ChangeReport:
    t1_scene_id: str
    t2_scene_id: str
    t1_timestamp: str | None = None
    t2_timestamp: str | None = None
    tiles_compared: int = 0
    tiles_skipped: int = 0
    elapsed_seconds: float = 0.0
    results: list[ChangeTile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "t1_scene_id": self.t1_scene_id,
            "t2_scene_id": self.t2_scene_id,
            "t1_timestamp": self.t1_timestamp,
            "t2_timestamp": self.t2_timestamp,
            "tiles_compared": self.tiles_compared,
            "tiles_skipped": self.tiles_skipped,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "results": [r.as_dict() for r in self.results],
            "errors": self.errors,
        }


def generate_batches(iterable, batch_size=8):
    """Yields batches of up to batch_size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _unusable(q: TileQuality) -> bool:
    """
    True when a chip is genuinely uninterpretable on this date.

    Only cloud and no-data disqualify a chip from a temporal comparison. Low
    variance does not - see the call site in compare_scenes().
    """
    return (
        q.cloud_fraction > settings.max_cloud_fraction
        or q.nodata_fraction > settings.max_nodata_fraction
    )


def _cells(reader: SceneReader, row: int | None, col: int | None):
    """Either one explicit cell or the whole grid, row-major."""
    rows, cols = reader.grid_shape()
    if row is not None and col is not None:
        yield row, col
        return
    for r in range(rows):
        for c in range(cols):
            yield r, c


def compare_scenes(
    t1_source: str,
    t2_source: str,
    *,
    t1_scene_id: str | None = None,
    t2_scene_id: str | None = None,
    row: int | None = None,
    col: int | None = None,
    min_change_score: float = 0.0,
    max_tiles: int | None = None,
    top_k: int | None = None,
    skip_cloudy: bool = True,
) -> ChangeReport:
    """
    Score temporal change for co-registered grid cells across two acquisitions.

    row/col together restrict the run to a single chip. min_change_score drops
    quiet tiles from the response; top_k truncates after sorting by score, so an
    analyst gets the strongest signals first instead of a whole-scene dump.
    """
    started = time.perf_counter()

    with SceneReader(t1_source, scene_id=t1_scene_id) as s1, \
         SceneReader(t2_source, scene_id=t2_scene_id) as s2:

        report = ChangeReport(
            t1_scene_id=s1.scene_id,
            t2_scene_id=s2.scene_id,
            t1_timestamp=s1.acquisition_timestamp,
            t2_timestamp=s2.acquisition_timestamp,
        )

        reason = s1.is_grid_compatible(s2)
        if reason is not None:
            raise RasterEngineError(
                f"{s1.scene_id} and {s2.scene_id} are not comparable: {reason}"
            )

        # Warm the model once so per-tile timing reflects inference, not load.
        prithvi.load()

        for batch in generate_batches(_cells(s1, row, col), batch_size=8):
            if max_tiles is not None and report.tiles_compared >= max_tiles:
                break
            
            valid_batch = []
            t1_stacks = []
            t2_stacks = []
            
            # Prepare the batch
            for r, c in batch:
                try:
                    q1 = s1.tile_quality(r, c)
                    q2 = s2.tile_quality(r, c)
                except Exception as exc:
                    report.errors.append(f"r{r}c{c}: quality probe failed: {exc}")
                    report.tiles_skipped += 1
                    continue

                if skip_cloudy and (_unusable(q1) or _unusable(q2)):
                    report.tiles_skipped += 1
                    continue
                    
                try:
                    # Normalization happens sequentially, but inference will be batched
                    t1_stacks.append(normalize_for_prithvi(s1.read_tile_stack(r, c)))
                    t2_stacks.append(normalize_for_prithvi(s2.read_tile_stack(r, c)))
                    valid_batch.append((r, c, q1, q2))
                except Exception as exc:
                    report.errors.append(f"r{r}c{c}: prep failed: {exc}")
                    report.tiles_skipped += 1
                    continue
                    
            if not valid_batch:
                continue
                
            # Execute Forward Pass on Batch with strict memory enforcement
            with torch.no_grad():
                results = prithvi.compare_batch(t1_stacks, t2_stacks)
            
            # Process results
            for (r, c, q1, q2), result in zip(valid_batch, results):
                report.tiles_compared += 1
                if result.change_score < min_change_score:
                    continue

                report.results.append(
                    ChangeTile(
                        tile_id=f"{s1.scene_id}_r{r:04d}c{c:04d}",
                        row=r,
                        col=c,
                        wgs84_bounding_box=list(s1.window_bounds_wgs84(s1.tile_window(r, c))),
                        change_score=result.change_score,
                        cosine_distance=result.cosine_distance,
                        l2_distance=result.l2_distance,
                        t1_cloud=round(q1.cloud_fraction, 4),
                        t2_cloud=round(q2.cloud_fraction, 4),
                    )
                )

            # Prevent VRAM Leaks
            torch.cuda.empty_cache()
            
            # Optional garbage collection if processing many batches
            if report.tiles_compared % 32 == 0:
                import gc
                gc.collect()

    report.results.sort(key=lambda t: t.change_score, reverse=True)
    if top_k is not None:
        report.results = report.results[:top_k]

    report.elapsed_seconds = time.perf_counter() - started
    logger.info(
        "Change %s vs %s: %d compared, %d skipped, %d above threshold in %.1fs",
        report.t1_scene_id, report.t2_scene_id, report.tiles_compared,
        report.tiles_skipped, len(report.results), report.elapsed_seconds,
    )
    return report


def to_geojson(report: ChangeReport) -> dict:
    """FeatureCollection of change tiles - the analyst's export format."""
    features = []
    for t in report.results:
        w, s, e, n = t.wgs84_bounding_box
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            },
            "properties": {
                "tile_id": t.tile_id,
                "change_score": t.change_score,
                "cosine_distance": t.cosine_distance,
                "l2_distance": t.l2_distance,
                "t1_scene_id": report.t1_scene_id,
                "t2_scene_id": report.t2_scene_id,
                "t1_timestamp": report.t1_timestamp,
                "t2_timestamp": report.t2_timestamp,
            },
        })
    return {
        "type": "FeatureCollection",
        "name": f"trinetra_change_{report.t1_scene_id}_vs_{report.t2_scene_id}",
        "features": features,
    }
