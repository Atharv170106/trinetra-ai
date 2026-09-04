/**
 * Natural-language retrieval controls.
 *
 * score_threshold is intentionally NOT exposed as a 0-1 slider. RemoteCLIP
 * text-image cosine scores cluster around 0.15-0.35, so a naive "minimum
 * confidence 50%" control would silently return nothing and read as a broken
 * search. The confidence slider maps onto that real range instead.
 */
import { useState } from "react";

const EXAMPLES = [
  "aircraft parked on a runway",
  "military vehicles in an open compound",
  "newly constructed buildings",
  "bridge over a river",
  "dense forest canopy",
];

export default function SearchPanel({ scenes, busy, onSearch }) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);
  const [threshold, setThreshold] = useState(0);
  const [sceneId, setSceneId] = useState("");
  const [maxCloud, setMaxCloud] = useState(100);

  const submit = (event) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || busy) return;
    onSearch({
      query: trimmed,
      limit: Number(limit),
      scoreThreshold: threshold > 0 ? threshold : null,
      sceneIds: sceneId ? [sceneId] : null,
      maxCloud: maxCloud < 100 ? maxCloud / 100 : null,
    });
  };

  return (
    <form className="panel-scroll" onSubmit={submit}>
      <div className="field">
        <label className="lbl" htmlFor="q">
          Target description
        </label>
        <textarea
          id="q"
          className="txt"
          rows={2}
          placeholder="e.g. aircraft parked on a runway"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            // Enter submits; Shift+Enter adds a line, matching chat conventions.
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
        />
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
          <span className="range-val">{maxCloud === 100 ? "any" : `${maxCloud}%`}</span>
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
