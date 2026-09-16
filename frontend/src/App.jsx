/**
 * Trinetra AI - analyst dashboard shell.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, downloadBlob } from "./api/client";
import LeafletMap from "./map/LeafletMap";
import ResultCard from "./components/ResultCard";
import OmniSearch from "./components/OmniSearch";
import ChangePanel from "./components/ChangePanel";
import IngestPanel from "./components/IngestPanel";
import MetadataDrawer from "./components/MetadataDrawer";
import TriageModal from "./components/TriageModal";
import IntelStatusCard from "./components/IntelStatusCard";

// Removed TABS array since we are eliminating sidebar tabs

export default function App() {
  const [health, setHealth] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [mode, setMode] = useState("search");
  const [hits, setHits] = useState([]);
  const [meta, setMeta] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [verdicts, setVerdicts] = useState({});
  const [ingestReport, setIngestReport] = useState(null);
  const [showImagery, setShowImagery] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [flash, setFlash] = useState(null);
  const [drawnBbox, setDrawnBbox] = useState(null);

  // Modal states for advanced tools
  const [showIngestModal, setShowIngestModal] = useState(false);
  const [showChangeModal, setShowChangeModal] = useState(false);

  const lastQuery = meta?.query ?? null;

  const refreshScenes = useCallback(async () => {
    try { setScenes(await api.scenes()); } catch (e) { setScenes([]); }
  }, []);

  const refreshVerdicts = useCallback(async () => {
    try {
      const log = await api.triageLog();
      setVerdicts(log.verdicts ?? {});
    } catch { setVerdicts({}); }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await api.health();
        if (!cancelled) setHealth(h);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    })();
    refreshScenes();
    refreshVerdicts();
    const timer = setInterval(async () => {
      try { setHealth(await api.health()); } catch { setHealth((prev) => (prev ? { ...prev, status: "degraded" } : null)); }
    }, 20000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [refreshScenes, refreshVerdicts]);

  useEffect(() => {
    if (!flash) return;
    const timer = setTimeout(() => setFlash(null), 4000);
    return () => clearTimeout(timer);
  }, [flash]);

  const run = async (fn) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  const handleSearch = (params) =>
    run(async () => {
      const res = await api.search(params);
      setMode("search");
      setHits(res.hits);
      setMeta({ query: res.query, count: res.count, elapsed: `${res.elapsed_ms} ms` });
      setSelectedId(res.hits[0]?.tile_id ?? null);
      if (res.count === 0) setFlash("No chips matched. Lower the similarity threshold or widen the filters.");
    });

  const handleCompare = (params) =>
    run(async () => {
      const res = await api.temporalChange(params);
      setMode("change");
      setHits(res.results);
      setMeta({
        query: `${res.t1_scene_id} → ${res.t2_scene_id}`,
        count: res.results.length,
        elapsed: `${res.elapsed_seconds} s`,
        compared: res.tiles_compared,
        skipped: res.tiles_skipped,
        t1_scene_id: res.t1_scene_id,
        t2_scene_id: res.t2_scene_id,
      });
      setSelectedId(res.results[0]?.tile_id ?? null);
      if (res.errors?.length) setFlash(`${res.errors.length} chip error(s): ${res.errors[0]}`);
      setShowChangeModal(false);
    });

  const handleIngest = (params) =>
    run(async () => {
      const report = await api.ingest(params);
      setIngestReport(report);
      await refreshScenes();
      setFlash(`Indexed ${report.tiles_indexed} chip(s) from ${report.scene_id} in ${report.elapsed_seconds}s.`);
    });

  const handleTriage = (tileId, verdict, note) =>
    run(async () => {
      await api.triage({ tileId, verdict, query: lastQuery, note: note || null });
      await refreshVerdicts();
      setFlash(`${tileId} marked ${verdict === "confirmed" ? "confirmed intel" : "false alarm"}.`);
    });

  const handleExport = () =>
    run(async () => {
      const tileIds = hits.map((h) => h.tile_id);
      if (!tileIds.length) throw new Error("Nothing to export — run a search first.");
      const blob = await api.exportReport({ tileIds, query: lastQuery });
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "");
      downloadBlob(blob, `trinetra_report_${stamp}.geojson`);
      setFlash(`Exported ${tileIds.length} tile(s) as GeoJSON.`);
    });

  const selectedHit = useMemo(() => hits.find((h) => h.tile_id === selectedId) ?? null, [hits, selectedId]);
  const triagedCount = useMemo(() => hits.filter((h) => verdicts[h.tile_id]).length, [hits, verdicts]);

  const gpu = health?.gpu;
  const qdrantOk = health?.qdrant?.reachable;
  const statusClass = health == null ? "" : health.status === "ok" ? "ok" : "warn";

  return (
    <div className="app">
      <header className="masthead">
        <h1>Trinetra <span>AI</span></h1>
        <div className="subtitle">Imagery Intelligence · Offline</div>
        <div className="spacer" />
        <div className="status-cluster">
          <span className="status-pill"><i className={`dot ${statusClass}`} />{health?.status ?? "connecting"}</span>
          <span className="status-pill"><i className={`dot ${qdrantOk ? "ok" : "bad"}`} />qdrant {health?.qdrant?.points != null ? `${health.qdrant.points} pts` : ""}</span>
          <span className="status-pill">
            <i className={`dot ${gpu?.cuda_available ? "ok" : "warn"}`} />
            {gpu?.cuda_available ? `${gpu.vram_allocated_gb?.toFixed(2)}/${gpu.vram_total_gb?.toFixed(1)} GB` : "cpu"}
          </span>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="panel">
            {error && <div className="notice error">{error}</div>}
            {flash && !error && <div className="notice good">{flash}</div>}

            <div className="panel-scroll" style={{ padding: "14px", color: "var(--text-1)" }}>
              <IntelStatusCard />
            </div>
          </div>

          {hits.length > 0 && (
            <>
              <div className="result-meta">
                <span>
                  {meta?.count} {mode === "change" ? "changed" : "hits"}
                  {meta?.compared != null ? ` / ${meta.compared} compared` : ""}
                  {triagedCount > 0 ? ` · ${triagedCount} triaged` : ""}
                </span>
                <span>{meta?.elapsed}</span>
              </div>
              <div className="panel-scroll" style={{ maxHeight: "45vh", flex: "none" }}>
                {hits.map((hit) => (
                  <ResultCard key={hit.tile_id} hit={hit} mode={mode} active={hit.tile_id === selectedId} verdict={verdicts[hit.tile_id]?.verdict} onSelect={setSelectedId} />
                ))}
              </div>
              <div className="drawer-foot" style={{ borderTop: "1px solid var(--border)" }}>
                <label className="status-pill" style={{ fontSize: 11, color: "var(--text-1)" }}>
                  <input type="checkbox" checked={showImagery} onChange={(e) => setShowImagery(e.target.checked)} />
                  Overlay chip imagery on map
                </label>
                <button className="btn block primary" onClick={handleExport} disabled={busy}>⬇ Export Intel Report (GeoJSON)</button>
              </div>
            </>
          )}
        </aside>

        <main className="map-region">
          <OmniSearch
            scenes={scenes}
            busy={busy}
            drawnBbox={drawnBbox}
            onSearch={handleSearch}
            onOpenIngest={() => setShowIngestModal(true)}
            onOpenChange={() => setShowChangeModal(true)}
          />
          <LeafletMap hits={hits} mode={mode} selectedId={selectedId} verdicts={verdicts} showImagery={showImagery} meta={meta} onSelect={setSelectedId} onBoundingBoxChange={setDrawnBbox} />
          {selectedHit && (
            <MetadataDrawer hit={selectedHit} mode={mode} verdict={verdicts[selectedHit.tile_id]} onClose={() => setSelectedId(null)} onTriage={handleTriage} />
          )}
          <TriageModal onAcknowledge={(alert) => setFlash(`Acknowledged alert for ${alert.scene}`)} onFalseAlarm={(alert) => setFlash(`Marked ${alert.scene} as false alarm.`)} />
        </main>
      </div>

      {/* ----------------------------- Overlays & Modals ----------------------------- */}
      
      {showIngestModal && (
        <div className="tool-modal-overlay">
          <div className="tool-modal-content tactical-glass">
            <div className="tool-modal-header">
              <h3>Ingest Satellite Data</h3>
              <button className="close-btn" onClick={() => setShowIngestModal(false)}>✕</button>
            </div>
            <div className="tool-modal-body">
              <IngestPanel scenes={scenes} busy={busy} report={ingestReport} onIngest={handleIngest} onRefresh={refreshScenes} />
            </div>
          </div>
        </div>
      )}

      {showChangeModal && (
        <div className="tool-modal-overlay">
          <div className="tool-modal-content tactical-glass wide">
            <div className="tool-modal-header">
              <h3>Temporal Change Detection</h3>
              <button className="close-btn" onClick={() => setShowChangeModal(false)}>✕</button>
            </div>
            <div className="tool-modal-body">
              <ChangePanel busy={busy} onCompare={handleCompare} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
