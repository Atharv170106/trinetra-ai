/**
 * Trinetra AI - Axios client for the FastAPI backend.
 *
 * Every call funnels through `unwrap` so a FastAPI `detail` string, a Pydantic
 * 422 validation array, or a dead backend all surface as one readable Error
 * message instead of an opaque "Request failed with status code 422".
 */
import axios from "axios";

// Same-origin in production (Phase 6 serves the built bundle from FastAPI);
// the Vite dev proxy handles localhost:8000 during development.
const http = axios.create({
  baseURL: "/api",
  // Ingest and whole-scene change runs are minutes long, not seconds.
  timeout: 30 * 60 * 1000,
  headers: { "Content-Type": "application/json" },
});

function describe(error) {
  if (error.response) {
    const { status, data } = error.response;
    // Pydantic 422 bodies are {detail: [{loc, msg, type}, ...]}.
    if (Array.isArray(data?.detail)) {
      const first = data.detail[0];
      const field = Array.isArray(first?.loc) ? first.loc.slice(1).join(".") : "input";
      return `${status}: ${field} — ${first?.msg ?? "invalid"}`;
    }
    if (typeof data?.detail === "string") return `${status}: ${data.detail}`;
    return `${status}: ${error.response.statusText || "request failed"}`;
  }
  if (error.code === "ECONNABORTED") return "Request timed out.";
  return "Cannot reach the Trinetra backend. Is it running on port 8080 (or 8000)?";
}

async function unwrap(promise) {
  try {
    const res = await promise;
    return res.data;
  } catch (error) {
    throw new Error(describe(error));
  }
}

export const api = {
  health: () => unwrap(http.get("/health")),
  scenes: () => unwrap(http.get("/scenes")),
  datasets: () => unwrap(http.get("/datasets")),

  search: ({ query, limit = 20, scoreThreshold = null, sceneIds = null, maxCloud = null, boundingBox = null, dateRange = null }) =>
    unwrap(
      http.post("/search", {
        query,
        limit,
        score_threshold: scoreThreshold,
        scene_ids: sceneIds,
        max_cloud: maxCloud,
        bounding_box: boundingBox,
        date_range: dateRange,
      })
    ),

  ingest: ({ source, sceneId = null, maxTiles = null, replace = true }) =>
    unwrap(
      http.post("/ingest", {
        source,
        scene_id: sceneId,
        max_tiles: maxTiles,
        replace,
        save_previews: true,
      })
    ),

  temporalChange: ({
    t1Source,
    t2Source,
    t1SceneId = null,
    t2SceneId = null,
    minChangeScore = 0,
    topK = 50,
    skipCloudy = true,
  }) =>
    unwrap(
      http.post("/temporal-change", {
        t1_source: t1Source,
        t2_source: t2Source,
        t1_scene_id: t1SceneId,
        t2_scene_id: t2SceneId,
        min_change_score: minChangeScore,
        top_k: topK,
        skip_cloudy: skipCloudy,
      })
    ),

  triage: ({ tileId, verdict, query = null, note = null }) =>
    unwrap(
      http.post("/triage", {
        tile_id: tileId,
        verdict,
        query,
        analyst_note: note,
      })
    ),

  triageLog: () => unwrap(http.get("/triage")),
  /** Get the bounding box from the most recently downloaded scene. */
  pipelineBbox: () => unwrap(http.get("/pipeline/bbox")),

  /** Trigger ingest_pipeline.py to download Sentinel-2 scenes from Element84. */
  pipelineIngest: ({ startDate, endDate, bbox, maxCloud = 20, limit = 2 }) =>
    unwrap(
      http.post("/pipeline/ingest", {
        start_date: startDate,
        end_date: endDate,
        bbox,
        max_cloud: maxCloud,
        limit,
      })
    ),

  /** Poll the status of a pipeline download job. */
  pipelineStatus: (jobId) => unwrap(http.get(`/pipeline/status/${jobId}`)),

  /** Returns a Blob so the caller can trigger a download without a server file. */
  exportReport: async ({ tileIds, query = null, includeUnverified = true }) => {
    try {
      const res = await http.post(
        "/export",
        { tile_ids: tileIds, query, include_unverified: includeUnverified },
        { responseType: "blob" }
      );
      return res.data;
    } catch (error) {
      // An error body arrives as a Blob too; read it back so the message is useful.
      if (error.response?.data instanceof Blob) {
        try {
          const text = await error.response.data.text();
          const parsed = JSON.parse(text);
          throw new Error(`${error.response.status}: ${parsed.detail ?? text}`);
        } catch (inner) {
          if (inner instanceof Error && inner.message.startsWith(`${error.response.status}:`)) {
            throw inner;
          }
        }
      }
      throw new Error(describe(error));
    }
  },

  setWatchdogAoi: (bbox) => unwrap(http.post("/watchdog/aoi", { bbox })),
};

export const tilePreviewUrl = (tileId) => `/api/tiles/${encodeURIComponent(tileId)}`;

/** Browser-side file save. No server round trip, so it works offline. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
