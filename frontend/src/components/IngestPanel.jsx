import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { api } from "../api/client";

export default function IngestPanel({ scenes, busy, report, onIngest, onRefresh }) {
  const [source, setSource] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [maxTiles, setMaxTiles] = useState("");
  const [dateFilter, setDateFilter] = useState("");
  
  const [datasets, setDatasets] = useState([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);

  // ---------- Pipeline download state ----------
  const [dlDate, setDlDate] = useState("");
  const [dlBbox, setDlBbox] = useState(null); // auto-loaded from backend
  const [dlJobId, setDlJobId] = useState(null);
  const [dlStatus, setDlStatus] = useState(null);
  const [dlMessage, setDlMessage] = useState("");
  const pollRef = useRef(null);

  // Auto-load bbox from existing scenes on mount
  useEffect(() => {
    async function loadBbox() {
      try {
        const res = await api.pipelineBbox();
        setDlBbox(res.bbox);
      } catch (err) {
        console.error("Failed to load bbox, using Jaisalmer default", err);
        setDlBbox([70.8, 26.8, 71.0, 27.0]);
      }
    }
    loadBbox();
  }, []);

  useEffect(() => {
    async function loadDatasets() {
      try {
        const data = await api.datasets();
        setDatasets(data);
        if (data.length > 0) {
          setSource(data[0].path);
        }
      } catch (err) {
        console.error("Failed to load datasets for IngestPanel", err);
      } finally {
        setLoadingDatasets(false);
      }
    }
    loadDatasets();
  }, []);

  const refreshDatasets = useCallback(async () => {
    try {
      const data = await api.datasets();
      setDatasets(data);
      if (data.length > 0 && !source) {
        setSource(data[0].path);
      }
    } catch (err) {
      console.error("Failed to refresh datasets", err);
    }
  }, [source]);

  const filteredDatasets = useMemo(() => {
    if (!dateFilter) return datasets;
    return datasets.filter(ds => ds.date === dateFilter);
  }, [datasets, dateFilter]);

  useEffect(() => {
    if (filteredDatasets.length > 0) {
      setSource(filteredDatasets[0].path);
    } else {
      setSource("");
    }
  }, [filteredDatasets]);

  // Poll pipeline job status
  useEffect(() => {
    if (!dlJobId || (dlStatus !== "accepted" && dlStatus !== "running")) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const res = await api.pipelineStatus(dlJobId);
        setDlStatus(res.status);
        setDlMessage(res.message);
        if (res.status === "completed" || res.status === "failed") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          if (res.status === "completed") {
            await refreshDatasets();
          }
        }
      } catch (err) {
        setDlMessage(`Poll error: ${err.message}`);
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [dlJobId, dlStatus, refreshDatasets]);

  const handleDownload = async () => {
    if (!dlDate) {
      setDlMessage("Please select a date.");
      return;
    }
    if (!dlBbox) {
      setDlMessage("Coordinates not loaded yet. Try again in a moment.");
      return;
    }
    setDlStatus("accepted");
    setDlMessage("Submitting download job...");
    try {
      const res = await api.pipelineIngest({
        startDate: dlDate,
        endDate: dlDate,
        bbox: dlBbox,
        maxCloud: 20,
        limit: 1,
      });
      setDlJobId(res.job_id);
      setDlStatus(res.status);
      setDlMessage(res.message);
    } catch (err) {
      setDlStatus("failed");
      setDlMessage(err.message);
    }
  };

  const submit = (event) => {
    event.preventDefault();
    if (!source || busy) return;
    onIngest({
      source: source,
      sceneId: sceneId.trim() || null,
      maxTiles: maxTiles ? Number(maxTiles) : null,
    });
  };

  const isPipelineBusy = dlStatus === "accepted" || dlStatus === "running";
  const bboxLabel = dlBbox
    ? `${dlBbox[0].toFixed(2)}°W, ${dlBbox[1].toFixed(2)}°S, ${dlBbox[2].toFixed(2)}°E, ${dlBbox[3].toFixed(2)}°N`
    : "Loading...";

  return (
    <div className="panel-scroll">
      {/* ===== Section 1: Download New Data by Date ===== */}
      <div style={{ marginBottom: "2rem", paddingBottom: "1.5rem", borderBottom: "1px solid var(--border)" }}>
        <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent)" }}>
          🛰️ Download New Satellite Data
        </h3>
        <div className="hint" style={{ marginBottom: "0.75rem" }}>
          Select a date to download Sentinel-2 imagery for the same region as your existing data.
        </div>

        <div className="field">
          <label className="lbl" htmlFor="dlDate">Acquisition Date</label>
          <input
            id="dlDate"
            className="txt"
            type="date"
            value={dlDate}
            onChange={(e) => setDlDate(e.target.value)}
            disabled={isPipelineBusy}
          />
        </div>

        <div className="field" style={{ marginTop: "0.75rem" }}>
          <label className="lbl">Region (auto-detected)</label>
          <div className="txt" style={{ opacity: 0.7, fontSize: "0.85rem", cursor: "default" }}>
            📍 {bboxLabel}
          </div>
          <div className="hint">
            Coordinates are extracted from your previously downloaded scenes.
          </div>
        </div>

        <div className="field" style={{ marginTop: "1rem" }}>
          <button
            className={`btn primary block ${isPipelineBusy ? "busy-pulse" : ""}`}
            onClick={handleDownload}
            disabled={isPipelineBusy || !dlDate || !dlBbox}
          >
            {isPipelineBusy ? "⏳ Downloading…" : "🔒 Initiate Secure Download"}
          </button>
        </div>

        {dlMessage && (
          <div
            className={`notice ${dlStatus === "failed" ? "error" : dlStatus === "completed" ? "ok" : "busy"}`}
            style={{ margin: "10px 0 0" }}
          >
            {dlStatus === "completed" && "✅ "}
            {dlStatus === "failed" && "❌ "}
            {dlMessage}
          </div>
        )}
      </div>

      {/* ===== Section 2: Ingest existing scene ===== */}
      <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent)" }}>
        📦 Ingest to Vector Index
      </h3>
      <form onSubmit={submit}>
        <div className="field">
          <label className="lbl" htmlFor="datePicker">
            Filter Data by Capture Date (Optional)
          </label>
          <input
            id="datePicker"
            className="txt"
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
          />
          <div className="hint">
            Select a date to automatically find all datasets dropped into the secure drop zone on that day.
          </div>
        </div>

        <div className="field" style={{ marginTop: "1rem" }}>
          <label className="lbl" htmlFor="src">
            Scene to Ingest
          </label>
          {loadingDatasets ? (
            <div className="txt">Scanning secure drop zone...</div>
          ) : (
            <select
              id="src"
              className="txt"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              disabled={filteredDatasets.length === 0}
            >
              {filteredDatasets.length === 0 ? (
                <option value="">No datasets found for this date.</option>
              ) : (
                filteredDatasets.map((ds) => (
                  <option key={`ingest-${ds.path}`} value={ds.path}>
                    {ds.date ? `[${ds.date}] ` : ""}{ds.name} ({ds.location})
                  </option>
                ))
              )}
            </select>
          )}
          <div className="hint">
            Files in the secure_drop_zone and sample_data are detected automatically.
          </div>
        </div>

        <div className="field" style={{ marginTop: "1rem" }}>
          <div className="row">
            <div>
              <label className="lbl" htmlFor="sid">
                Scene ID (Optional Override)
              </label>
              <input
                id="sid"
                className="txt"
                placeholder="auto"
                value={sceneId}
                onChange={(e) => setSceneId(e.target.value)}
              />
            </div>
            <div>
              <label className="lbl" htmlFor="mt">
                Max tiles
              </label>
              <input
                id="mt"
                className="txt"
                type="number"
                min="1"
                placeholder="all"
                value={maxTiles}
                onChange={(e) => setMaxTiles(e.target.value)}
              />
            </div>
          </div>
          <div className="hint">
            Set max tiles to 50 for a fast smoke test before committing to a full
            scene.
          </div>
        </div>

        <div className="field" style={{ marginTop: "1.5rem" }}>
          <button
            className={`btn primary block ${busy ? "busy-pulse" : ""}`}
            type="submit"
            disabled={busy || !source}
          >
            {busy ? "Indexing…" : "Ingest scene"}
          </button>
          {busy && (
            <div className="notice busy" style={{ margin: "10px 0 0" }}>
              Tiling, embedding, and indexing. A full Sentinel-2 tile is roughly
              1800 chips — keep this tab open.
            </div>
          )}
        </div>
      </form>

      {report && (
        <div className="field" style={{ marginTop: "2rem" }}>
          <label className="lbl">Last ingest</label>
          <dl className="kv">
            <dt>Scene</dt>
            <dd>{report.scene_id}</dd>
            <dt>Indexed</dt>
            <dd>
              {report.tiles_indexed} of {report.tiles_seen}
            </dd>
            <dt>Speed</dt>
            <dd>
              {report.elapsed_seconds.toFixed(1)}s (
              {report.tiles_per_second.toFixed(1)}/s)
            </dd>
          </dl>
          {report.errors?.length > 0 && (
            <div className="notice error">
              {report.errors.length} errors, see server logs for details.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
