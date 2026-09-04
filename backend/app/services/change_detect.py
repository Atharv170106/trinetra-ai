"""
Trinetra AI - Phase 4: Multi-temporal change analysis.

Nothing is persisted. Two scenes are opened side by side, the same grid cells
are read from both, and Prithvi scores each pair on demand. Only one chip pair
is resident at a time, so a full-scene diff costs the same RAM as a single tile.

Grid compatibility is checked up front: comparing r,c across two scenes only
means anything if both share a CRS, size, and geotransform.
"""

from __future__ import annotations

import gc
import logging
import time
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

        for r, c in _cells(s1, row, col):
            if max_tiles is not None and report.tiles_compared >= max_tiles:
                break

            try:
                q1 = s1.tile_quality(r, c)
                q2 = s2.tile_quality(r, c)
            except Exception as exc:
                report.errors.append(f"r{r}c{c}: quality probe failed: {exc}")
                report.tiles_skipped += 1
                continue

            # A cloud or a data gap on either date makes the comparison
            # meaningless. Deliberately NOT using quality.rejected: that also
            # trips the min_stddev flat-chip test, which exists to keep
            # featureless chips out of the retrieval index. A uniform chip is a
            # perfectly valid change target - a new concrete pad or a cleared
            # airstrip reads as flat, and skipping it would hide the signal.
            if skip_cloudy and (_unusable(q1) or _unusable(q2)):
                report.tiles_skipped += 1
                continue

            stack1 = stack2 = None
            try:
                stack1 = normalize_for_prithvi(s1.read_tile_stack(r, c))
                stack2 = normalize_for_prithvi(s2.read_tile_stack(r, c))
                result = prithvi.compare(stack1, stack2)
            except (InferenceError, ValueError, RasterEngineError) as exc:
                report.errors.append(f"r{r}c{c}: {exc}")
                report.tiles_skipped += 1
                continue
            finally:
                stack1 = stack2 = None
                if report.tiles_compared % 32 == 0:
                    gc.collect()

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
