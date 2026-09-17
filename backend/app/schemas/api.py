"""
Trinetra AI - Phase 4: Request/response contracts.

Validation happens here so route handlers never guard against bad input, and
the OpenAPI schema at /docs doubles as the frontend's reference.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings

class WatchdogAOIRequest(BaseModel):
    bbox: list[float] | None = Field(
        None, min_length=4, max_length=4,
        description="GeoJSON bounding box [min_lon, min_lat, max_lon, max_lat] or null to clear."
    )


# ------------------------------------------------------------------- ingest
class IngestRequest(BaseModel):
    source: str = Field(
        ...,
        description="Path to a .SAFE directory, a per-band directory, or a "
                    "stacked GeoTIFF. Resolved on the server, not uploaded.",
    )
    scene_id: str | None = Field(
        None, description="Overrides the directory/file stem as the scene key."
    )
    max_tiles: int | None = Field(
        None, ge=1, description="Cap tiles indexed - useful for a quick smoke test."
    )
    replace: bool = Field(
        True, description="Drop this scene's existing points before indexing."
    )
    save_previews: bool = Field(
        True, description="Write a PNG per tile for the map overlay."
    )


class IngestResponse(BaseModel):
    scene_id: str
    tiles_seen: int
    tiles_rejected: int
    tiles_indexed: int
    elapsed_seconds: float
    tiles_per_second: float
    errors: list[str] = []


# ------------------------------------------------------------------- search
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       description="Natural-language description of the target.")
    limit: int = Field(20, ge=1)
    score_threshold: float | None = Field(
        None, ge=-1.0, le=1.0,
        description="Minimum cosine score. RemoteCLIP text-image scores are "
                    "typically 0.15-0.35; 1.0 would return nothing.",
    )
    scene_ids: list[str] | None = None
    bounding_box: list[float] | None = Field(
        None, min_length=4, max_length=4,
        description="GeoJSON bounding box [min_lon, min_lat, max_lon, max_lat]."
    )
    date_range: list[str] | None = Field(
        None, min_length=2, max_length=2,
        description="ISO8601 date range [start_date, end_date]."
    )
    max_cloud: float | None = Field(
        None, ge=0.0, le=1.0, description="Reject hits cloudier than this fraction."
    )

    @field_validator("limit")
    @classmethod
    def _cap_limit(cls, v: int) -> int:
        if v > settings.search_limit_max:
            raise ValueError(f"limit exceeds server maximum {settings.search_limit_max}")
        return v

    @field_validator("scene_ids")
    @classmethod
    def _no_empty_list(cls, v: list[str] | None) -> list[str] | None:
        # An empty list would build a MatchAny that matches nothing, which reads
        # as "no results" rather than "no filter".
        return v or None


class SearchHitModel(BaseModel):
    tile_id: str
    score: float
    scene_id: str | None = None
    acquisition_timestamp: str | None = None
    wgs84_bounding_box: list[float] | None = None
    scl_cloud_coverage: float | None = None
    row: int | None = None
    col: int | None = None
    preview_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    elapsed_ms: float
    hits: list[SearchHitModel]


# ----------------------------------------------------------- temporal change
class TemporalChangeRequest(BaseModel):
    t1_source: str = Field(..., description="Earlier acquisition.")
    t2_source: str = Field(..., description="Later acquisition.")
    t1_scene_id: str | None = None
    t2_scene_id: str | None = None
    row: int | None = Field(None, ge=0, description="With col, compares one chip only.")
    col: int | None = Field(None, ge=0)
    min_change_score: float = Field(0.0, ge=0.0, le=1.0)
    max_tiles: int | None = Field(None, ge=1)
    top_k: int | None = Field(None, ge=1)
    skip_cloudy: bool = True

    @model_validator(mode="after")
    def _row_col_together(self) -> "TemporalChangeRequest":
        if (self.row is None) != (self.col is None):
            raise ValueError("row and col must be supplied together")
        if self.top_k is not None and self.top_k > settings.change_top_k_max:
            raise ValueError(f"top_k exceeds server maximum {settings.change_top_k_max}")
        return self


class ChangeTileModel(BaseModel):
    tile_id: str
    row: int
    col: int
    wgs84_bounding_box: list[float]
    change_score: float
    cosine_distance: float
    l2_distance: float
    t1_cloud: float
    t2_cloud: float


class TemporalChangeResponse(BaseModel):
    t1_scene_id: str
    t2_scene_id: str
    t1_timestamp: str | None = None
    t2_timestamp: str | None = None
    tiles_compared: int
    tiles_skipped: int
    elapsed_seconds: float
    results: list[ChangeTileModel]
    errors: list[str] = []


# -------------------------------------------------------------- triage / audit
class TriageRequest(BaseModel):
    tile_id: str = Field(..., min_length=1)
    verdict: Literal["confirmed", "false_alarm"]
    query: str | None = Field(None, description="The query that surfaced this tile.")
    analyst_note: str | None = Field(None, max_length=2000)


class TriageResponse(BaseModel):
    tile_id: str
    verdict: str
    logged_at: str
    total_entries: int


class ExportRequest(BaseModel):
    tile_ids: list[str] = Field(..., min_length=1)
    query: str | None = None
    include_unverified: bool = True


# ---------------------------------------------------------------- diagnostics
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    qdrant: dict[str, Any]
    gpu: dict[str, Any]
    weights: dict[str, bool]


class SceneSummary(BaseModel):
    scene_id: str
    tile_count: int
    acquisition_timestamp: str | None = None
    wgs84_bounding_box: list[float] | None = None

class DatasetSummary(BaseModel):
    name: str
    path: str
    date: str | None = None
    location: str


# --------------------------------------------------------- pipeline ingest
class PipelineIngestRequest(BaseModel):
    start_date: str = Field(
        ..., description="Start date in YYYY-MM-DD format.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    end_date: str = Field(
        ..., description="End date in YYYY-MM-DD format.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box [West, South, East, North] in WGS84 degrees.",
    )
    max_cloud: float = Field(
        20.0, ge=0.0, le=100.0,
        description="Maximum cloud cover percentage (default 20).",
    )
    limit: int = Field(
        2, ge=1, le=10,
        description="Number of scenes to download (default 2).",
    )

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(cls, v: list[float]) -> list[float]:
        w, s, e, n = v
        if not (-180 <= w < e <= 180) or not (-90 <= s < n <= 90):
            raise ValueError("bbox must be [W, S, E, N] with W<E and S<N")
        return v


class PipelineIngestResponse(BaseModel):
    status: str = Field(..., description="accepted | running | completed | failed")
    message: str
    job_id: str


# ------------------------------------------------------------------- explain
class ExplainRequest(BaseModel):
    query: str = Field(..., description="The user's original search query or context")
    target_tile_id: str = Field(..., description="The primary tile ID to explain")
    baseline_tile_id: str | None = Field(None, description="The baseline tile ID for change analysis")

class ExplainResponse(BaseModel):
    summary: str
