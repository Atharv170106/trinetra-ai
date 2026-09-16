/**
 * Natural-language retrieval controls.
 * Omni-search with regex detection and Geoman integration.
 */
import { useState, useEffect } from "react";

const EXAMPLES = [
  "aircraft parked on a runway",
  "military vehicles in an open compound",
  "newly constructed buildings",
  "bridge over a river",
  "dense forest canopy",
];

export default function SearchPanel({ scenes, busy, drawnBbox, onSearch }) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);
  const [threshold, setThreshold] = useState(0);
  const [sceneId, setSceneId] = useState("");
  const [maxCloud, setMaxCloud] = useState(100);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  useEffect(() => {
    if (drawnBbox) {
      setQuery(`[${drawnBbox.join(", ")}]`);
    } else if (query.startsWith("[")) {
      setQuery("");
    }
  }, [drawnBbox]);

  const submit = (event) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || busy) return;

    // Detect if query is purely coordinates
    const isCoords = /^\[?\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]?$/.test(trimmed);
    
    let textQuery = trimmed;
    let boundingBox = drawnBbox;

    if (isCoords) {
      textQuery = "satellite imagery"; // Default fallback text
      try {
        boundingBox = JSON.parse(trimmed.replace(/^\[?/, "[").replace(/\]?$/, "]"));
      } catch(e) {}
    }

    const dateRange = (startDate && endDate) ? [
      new Date(startDate).toISOString(), 
      new Date(endDate).toISOString()
    ] : null;

    onSearch({
      query: textQuery,
      limit: Number(limit),
      scoreThreshold: threshold > 0 ? threshold : null,
      sceneIds: sceneId ? [sceneId] : null,
      maxCloud: maxCloud < 100 ? maxCloud / 100 : null,
      boundingBox: boundingBox,
      dateRange: dateRange,
    });
  };

  return (
    <form className="panel-scroll" onSubmit={submit}>
      <div className="field">
        <label className="lbl" htmlFor="q">
          Omni-Search (Text, Coords)
        </label>
        <div style={{ position: "relative" }}>
          <textarea
            id="q"
            className="txt"
            rows={2}
            placeholder="e.g. aircraft parked on a runway, or [W, S, E, N]"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) submit(e);
            }}
          />
          {/* Mock image upload icon as requested by the prompt */}
          <span style={{ position: "absolute", bottom: "10px", right: "10px", cursor: "pointer", opacity: 0.5 }} title="Upload Image (UI Mock)">
            📸
          </span>
        </div>
        <div className="hint">
          Try:{" "}
          {EXAMPLES.map((ex, i) => (
            <span key={ex}>
              {i > 0 && " · "}
              <a
                href="#"
                style={{ color: "var(--cyan)", textDecoration: "none" }}
                onClick={(e) => {
                  e.preventDefault();
                  setQuery(ex);
                }}
              >
                {ex}
              </a>
            </span>
          ))}
        </div>
      </div>

      <div className="field">
        <div className="row">
          <div>
            <label className="lbl">Start Date</label>
            <input type="date" className="txt" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="lbl">End Date</label>
            <input type="date" className="txt" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
        </div>
      </div>

      <div className="field">
        <div className="row">
          <div>
            <label className="lbl" htmlFor="limit">
              Top K
            </label>
            <select
              id="limit"
              className="txt"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            >
              {[10, 20, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="lbl" htmlFor="scene">
              Scene
            </label>
            <select
              id="scene"
              className="txt"
              value={sceneId}
              onChange={(e) => setSceneId(e.target.value)}
            >
              <option value="">All scenes</option>
              {scenes.map((s) => (
                <option key={s.scene_id} value={s.scene_id}>
                  {s.scene_id} ({s.tile_count})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="field">
        <label className="lbl" htmlFor="thr">
          Minimum similarity
        </label>
        <div className="range-row">
          <input
            id="thr"
            type="range"
            min="0"
            max="0.4"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <span className="range-val">{threshold === 0 ? "off" : threshold.toFixed(2)}</span>
        </div>
        <div className="hint">
          Cosine similarity. RemoteCLIP text-image scores typically fall between
          0.15 and 0.35, so anything above 0.40 returns nothing.
        </div>
      </div>

      <div className="field">
        <label className="lbl" htmlFor="cloud">
          Maximum cloud cover
        </label>
        <div className="range-row">
          <input
            id="cloud"
            type="range"
            min="0"
            max="100"
            step="5"
            value={maxCloud}
            onChange={(e) => setMaxCloud(Number(e.target.value))}
          />
          <span className="range-val">{maxCloud === 100 ? "off" : `${maxCloud}%`}</span>
        </div>
      </div>

      <div className="field">
        <button className="btn primary block" type="submit" disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search imagery"}
        </button>
      </div>
    </form>
  );
}
