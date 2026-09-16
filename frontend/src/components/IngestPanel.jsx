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
  const [dlStartDate, setDlStartDate] = useState("");
  const [dlEndDate, setDlEndDate] = useState("");
  const [dlBbox, setDlBbox] = useState({ w: "70.8", s: "26.8", e: "71.0", n: "27.0" });
  const [dlMaxCloud, setDlMaxCloud] = useState(20);
  const [dlLimit, setDlLimit] = useState(2);
  const [dlJobId, setDlJobId] = useState(null);
  const [dlStatus, setDlStatus] = useState(null); // null | "accepted" | "running" | "completed" | "failed"
  const [dlMessage, setDlMessage] = useState("");
  const pollRef = useRef(null);

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

  // When date filter changes, auto-select the first matching dataset
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
            // Auto-refresh datasets so the new scene appears in the dropdown
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
    const bbox = [
      parseFloat(dlBbox.w), parseFloat(dlBbox.s),
      parseFloat(dlBbox.e), parseFloat(dlBbox.n),
    ];
    if (bbox.some(isNaN)) {
      setDlMessage("Invalid bounding box values.");
      return;
    }
    if (!dlStartDate || !dlEndDate) {
      setDlMessage("Please select both start and end dates.");
      return;
    }
    setDlStatus("accepted");
    setDlMessage("Submitting pipeline job...");
    try {
      const res = await api.pipelineIngest({
        startDate: dlStartDate,
        endDate: dlEndDate,
        bbox,
        maxCloud: dlMaxCloud,
        limit: dlLimit,
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

  return (
    <div className="panel-scroll">
      {/* ===== Section 1: Secure Download ===== */}
      <div style={{ marginBottom: "2rem", paddingBottom: "1.5rem", borderBottom: "1px solid var(--border)" }}>
        <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent)" }}>
          🛰️ Secure Download
        </h3>
        <div className="hint" style={{ marginBottom: "1rem" }}>
          Pull Sentinel-2 L2A imagery from Element84 directly into the secure drop zone.
        </div>

        <div className="row" style={{ gap: "0.5rem", marginBottom: "0.75rem" }}>
          <div style={{ flex: 1 }}>
            <label className="lbl" htmlFor="dlStart">Start Date</label>
            <input id="dlStart" className="txt" type="date" value={dlStartDate}
              onChange={(e) => setDlStartDate(e.target.value)} disabled={isPipelineBusy} />
          </div>
          <div style={{ flex: 1 }}>
            <label className="lbl" htmlFor="dlEnd">End Date</label>
            <input id="dlEnd" className="txt" type="date" value={dlEndDate}
              onChange={(e) => setDlEndDate(e.target.value)} disabled={isPipelineBusy} />
          </div>
        </div>

        <label className="lbl">Bounding Box (W, S, E, N)</label>
        <div className="row" style={{ gap: "0.35rem", marginBottom: "0.75rem" }}>
          {["w", "s", "e", "n"].map((k) => (
            <input key={k} className="txt" type="number" step="0.01"
              style={{ flex: 1, minWidth: 0 }}
              placeholder={k.toUpperCase()}
              value={dlBbox[k]}
              onChange={(e) => setDlBbox(prev => ({ ...prev, [k]: e.target.value }))}
              disabled={isPipelineBusy} />
          ))}
        </div>

        <div className="row" style={{ gap: "0.5rem", marginBottom: "1rem" }}>
          <div style={{ flex: 1 }}>
            <label className="lbl" htmlFor="dlCloud">Max Cloud %</label>
            <input id="dlCloud" className="txt" type="number" min="0" max="100"
              value={dlMaxCloud} onChange={(e) => setDlMaxCloud(Number(e.target.value))}
              disabled={isPipelineBusy} />
          </div>
          <div style={{ flex: 1 }}>
            <label className="lbl" htmlFor="dlLimit">Scenes</label>
            <input id="dlLimit" className="txt" type="number" min="1" max="10"
              value={dlLimit} onChange={(e) => setDlLimit(Number(e.target.value))}
              disabled={isPipelineBusy} />
          </div>
        </div>

        <button
          className={`btn primary block ${isPipelineBusy ? "busy-pulse" : ""}`}
          onClick={handleDownload}
          disabled={isPipelineBusy || !dlStartDate || !dlEndDate}
        >
          {isPipelineBusy ? "⏳ Downloading…" : "🔒 Initiate Secure Download"}
        </button>

        {dlMessage && (
          <div className={`notice ${dlStatus === "failed" ? "error" : dlStatus === "completed" ? "ok" : "busy"}`}
            style={{ margin: "10px 0 0" }}>
            {dlStatus === "completed" && "✅ "}{dlStatus === "failed" && "❌ "}{dlMessage}
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
