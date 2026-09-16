import { useState, useEffect } from "react";
import { api } from "../api/client";

export default function ChangePanel({ busy, onCompare }) {
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [minScore, setMinScore] = useState(0);
  const [topK, setTopK] = useState(50);
  const [skipCloudy, setSkipCloudy] = useState(true);
  const [datasets, setDatasets] = useState([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);

  useEffect(() => {
    async function loadDatasets() {
      try {
        const data = await api.datasets();
        setDatasets(data);
        if (data.length > 0) {
          if (data.length > 1) {
            setT1(data[1].path);
            setT2(data[0].path);
          } else {
            setT1(data[0].path);
            setT2(data[0].path);
          }
        }
      } catch (err) {
        console.error("Failed to load datasets for ChangePanel", err);
      } finally {
        setLoadingDatasets(false);
      }
    }
    loadDatasets();
  }, []);

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
        {loadingDatasets ? (
          <div className="txt">Loading available datasets...</div>
        ) : (
          <select
            id="t1"
            className="txt"
            value={t1}
            onChange={(e) => setT1(e.target.value)}
          >
            {datasets.map((ds) => (
              <option key={`t1-${ds.path}`} value={ds.path}>
                {ds.date ? `[${ds.date}] ` : ""}{ds.name}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="field">
        <label className="lbl" htmlFor="t2">
          T2 — later acquisition
        </label>
        {loadingDatasets ? (
          <div className="txt">Loading available datasets...</div>
        ) : (
          <select
            id="t2"
            className="txt"
            value={t2}
            onChange={(e) => setT2(e.target.value)}
          >
            {datasets.map((ds) => (
              <option key={`t2-${ds.path}`} value={ds.path}>
                {ds.date ? `[${ds.date}] ` : ""}{ds.name}
              </option>
            ))}
          </select>
        )}
        <div className="hint">
          Available datasets in the secure drop zone and sample data. Both
          scenes must share a CRS, size, and geotransform.
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
            className="tactical-slider"
          />
          <span className="range-val" style={{ marginLeft: "12px" }}>{minScore === 0 ? "off" : minScore.toFixed(3)}</span>
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
              {[5, 10, 25, 50].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="lbl" htmlFor="cloudy">
              Cloudy chips
            </label>
            <select
              id="cloudy"
              className="txt"
              value={skipCloudy ? "skip" : "include"}
              onChange={(e) => setSkipCloudy(e.target.value === "skip")}
            >
              <option value="skip">Skip</option>
              <option value="include">Include</option>
            </select>
          </div>
        </div>
      </div>

      <div className="field" style={{ marginTop: "1rem" }}>
        <button
          className={`btn primary block ${busy ? "busy-pulse" : ""}`}
          type="submit"
          disabled={busy || !t1 || !t2}
        >
          {busy ? "Running Analysis…" : "Run change analysis"}
        </button>
        {busy && (
          <div className="notice busy" style={{ margin: "10px 0 0" }}>
            Whole-scene comparison runs Prithvi on every chip pair and can take
            several minutes. Nothing is indexed — results are computed on demand.
          </div>
        )}
      </div>
    </form>
  );
}
