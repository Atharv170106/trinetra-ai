/**
 * Multi-temporal change controls.
 *
 * Sources are server-side paths, not uploads: the scenes are gigabytes and
 * already sit on the host. Both must be co-registered — the backend returns 422
 * if they are not, which is surfaced verbatim rather than swallowed.
 */
import { useState } from "react";

export default function ChangePanel({ busy, onCompare }) {
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [minScore, setMinScore] = useState(0);
  const [topK, setTopK] = useState(50);
  const [skipCloudy, setSkipCloudy] = useState(true);

  const submit = (event) => {
    event.preventDefault();
    if (!t1.trim() || !t2.trim() || busy) return;
    onCompare({
      t1Source: t1.trim(),
      t2Source: t2.trim(),
      minChangeScore: minScore,
      topK: Number(topK),
      skipCloudy,
    });
  };

  return (
    <form className="panel-scroll" onSubmit={submit}>
      <div className="field">
        <label className="lbl" htmlFor="t1">
          T1 — earlier acquisition
        </label>
        <input
          id="t1"
          className="txt"
          placeholder="scene_2024_01 or an absolute path"
          value={t1}
          onChange={(e) => setT1(e.target.value)}
        />
      </div>

      <div className="field">
        <label className="lbl" htmlFor="t2">
          T2 — later acquisition
        </label>
        <input
          id="t2"
          className="txt"
          placeholder="scene_2026_01 or an absolute path"
          value={t2}
          onChange={(e) => setT2(e.target.value)}
        />
        <div className="hint">
          Relative names resolve inside <code>backend/sample_data/</code>. Both
          scenes must share a CRS, size, and geotransform — the comparison is
          refused otherwise, since chip r,c would not be the same ground.
        </div>
      </div>

      <div className="field">
        <label className="lbl" htmlFor="minscore">
          Minimum change score
        </label>
        <div className="range-row">
          <input
            id="minscore"
            type="range"
            min="0"
            max="0.2"
            step="0.005"
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
          />
          <span className="range-val">{minScore === 0 ? "off" : minScore.toFixed(3)}</span>
        </div>
        <div className="hint">
          Prithvi cosine distance between T1 and T2 features. Unchanged terrain
          scores near 0.01; raise this to isolate strong signals.
        </div>
      </div>

      <div className="field">
        <div className="row">
          <div>
            <label className="lbl" htmlFor="topk">
              Return top
            </label>
            <select
              id="topk"
              className="txt"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            >
              {[10, 25, 50, 100, 200].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="lbl" htmlFor="skip">
              Cloudy chips
            </label>
            <select
              id="skip"
              className="txt"
              value={skipCloudy ? "skip" : "keep"}
              onChange={(e) => setSkipCloudy(e.target.value === "skip")}
            >
              <option value="skip">Skip</option>
              <option value="keep">Compare anyway</option>
            </select>
          </div>
        </div>
      </div>

      <div className="field">
        <button
          className="btn primary block"
          type="submit"
          disabled={busy || !t1.trim() || !t2.trim()}
        >
          {busy ? "Comparing…" : "Run change analysis"}
        </button>
        <div className="hint">
          Whole-scene comparison runs Prithvi on every chip pair and can take
          several minutes. Nothing is indexed — results are computed on demand.
        </div>
      </div>
    </form>
  );
}
