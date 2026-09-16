import { useState, useEffect, useRef } from "react";
import TacticalFilters from "./TacticalFilters";
import { api as client } from "../api/client";

const QUICK_PROMPTS = [
  "aircraft on tarmac",
  "military vehicles in open compound",
  "newly constructed bunkers",
  "patrol boats near dock",
];

export default function OmniSearch({ scenes = [], busy, drawnBbox, onSearch, onOpenIngest, onOpenChange }) {
  const [query, setQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [limit, setLimit] = useState(20);
  const [threshold, setThreshold] = useState(0);
  const [sceneId, setSceneId] = useState("");
  const [maxCloud, setMaxCloud] = useState(100);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [detectedType, setDetectedType] = useState("SEMANTIC");
  const [isWatchdogArmed, setIsWatchdogArmed] = useState(false);
  const popoverRef = useRef(null);

  useEffect(() => {
    if (drawnBbox) {
      setQuery(`[${drawnBbox.join(", ")}]`);
      setDetectedType("AOI BOUNDS");
    } else if (query.startsWith("[")) {
      setQuery("");
      setDetectedType("SEMANTIC");
      if (isWatchdogArmed) handleToggleWatchdog(false);
    }
  }, [drawnBbox]);

  const handleToggleWatchdog = async (forceState) => {
    const nextState = forceState !== undefined ? forceState : !isWatchdogArmed;
    try {
      await client.setWatchdogAoi(nextState ? drawnBbox : null);
      setIsWatchdogArmed(nextState);
    } catch (err) {
      console.error("Failed to set watchdog AOI:", err);
    }
  };

  useEffect(() => {
    const trimmed = query.trim();
    if (/^\[?\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]?$/.test(trimmed)) {
      setDetectedType("COORDINATES");
    } else if (drawnBbox && query.startsWith("[")) {
      setDetectedType("AOI BOUNDS");
    } else {
      setDetectedType("SEMANTIC");
    }
  }, [query, drawnBbox]);

  useEffect(() => {
    function handleClickOutside(e) {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setShowFilters(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleResetFilters = () => {
    setStartDate("");
    setEndDate("");
    setSceneId("");
    setThreshold(0);
    setMaxCloud(100);
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || busy) return;

    let textQuery = trimmed;
    let boundingBox = drawnBbox;

    if (detectedType === "COORDINATES") {
      textQuery = "satellite terrain features";
      try {
        boundingBox = JSON.parse(trimmed.replace(/^\[?/, "[").replace(/\]?$/, "]"));
      } catch (err) {}
    }

    const dateRange =
      startDate && endDate
        ? [new Date(startDate).toISOString(), new Date(endDate).toISOString()]
        : null;

    onSearch({
      query: textQuery,
      limit: Number(limit),
      scoreThreshold: threshold > 0 ? threshold : null,
      sceneIds: sceneId ? [sceneId] : null,
      maxCloud: maxCloud < 100 ? maxCloud / 100 : null,
      boundingBox,
      dateRange,
    });
  };

  const hasActiveFilters = Boolean(startDate || endDate || sceneId || threshold > 0 || maxCloud < 100);

  return (
    <div className="omni-container" ref={popoverRef}>
      <form className="omni-bar tactical-glass" onSubmit={handleSubmit}>
        <div className="omni-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>

        <input
          type="text"
          className="omni-input"
          placeholder="Search terrain targets, or enter [W, S, E, N] coordinates..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="omni-badges">
          <span className={`omni-pill ${detectedType.toLowerCase().replace(" ", "-")}`}>
            {detectedType}
          </span>
        </div>

        <div className="omni-actions">
          <button
            type="button"
            className={`omni-action-btn ${isWatchdogArmed ? "active" : ""}`}
            title="Arm Air-Gapped AOI Watchdog"
            onClick={() => handleToggleWatchdog()}
            disabled={!drawnBbox}
            style={isWatchdogArmed ? { color: "var(--green)" } : {}}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
          </button>

          <button
            type="button"
            className="omni-action-btn"
            title="Temporal Change Detection (Prithvi-EO)"
            onClick={onOpenChange}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
            </svg>
          </button>

          <button
            type="button"
            className="omni-action-btn"
            title="Ingest Satellite Data"
            onClick={onOpenIngest}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </button>

          <button
            type="button"
            className={`omni-action-btn filter-btn ${showFilters || hasActiveFilters ? "active" : ""}`}
            title="Tactical Filters (Dates, Sensor, Clouds)"
            onClick={() => setShowFilters(!showFilters)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
            </svg>
            {hasActiveFilters && <span className="filter-active-dot" />}
          </button>

          <button type="submit" className={`omni-submit-btn ${busy ? "busy-pulse" : ""}`} disabled={busy || !query.trim()}>
            {busy ? "SCANNING" : "SCAN"}
          </button>
        </div>
      </form>

      <div className="quick-prompts">
        <span className="quick-label">PROMPTS:</span>
        {QUICK_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            className="quick-chip"
            onClick={() => setQuery(p)}
          >
            {p}
          </button>
        ))}
      </div>

      <TacticalFilters
        show={showFilters}
        onClose={() => setShowFilters(false)}
        scenes={scenes}
        startDate={startDate}
        setStartDate={setStartDate}
        endDate={endDate}
        setEndDate={setEndDate}
        sceneId={sceneId}
        setSceneId={setSceneId}
        limit={limit}
        setLimit={setLimit}
        threshold={threshold}
        setThreshold={setThreshold}
        maxCloud={maxCloud}
        setMaxCloud={setMaxCloud}
        onApply={() => { setShowFilters(false); handleSubmit(); }}
        onReset={handleResetFilters}
      />
    </div>
  );
}
