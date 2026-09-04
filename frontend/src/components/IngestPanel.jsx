/**
 * Scene ingest. Synchronous by design on the backend, so the form stays locked
 * with a live warning while a full tile indexes.
 */
import { useState } from "react";

export default function IngestPanel({ scenes, busy, report, onIngest, onRefresh }) {
  const [source, setSource] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [maxTiles, setMaxTiles] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (!source.trim() || busy) return;
    onIngest({
      source: source.trim(),
      sceneId: sceneId.trim() || null,
      maxTiles: maxTiles ? Number(maxTiles) : null,
    });
  };

  return (
    <div className="panel-scroll">
      <form onSubmit={submit}>
        <div className="field">
          <label className="lbl" htmlFor="src">
            Scene source
          </label>
          <input
            id="src"
            className="txt"
            placeholder="my_scene  ·  or C:\\data\\S2A_....SAFE"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          />
          <div className="hint">
            A .SAFE product directory, a folder of per-band GeoTIFFs (B02…B07 +
            SCL), or a stacked GeoTIFF. Relative names resolve inside{" "}
            <code>backend/sample_data/</code>. The file stays on the host — nothing
            is uploaded.
          </div>
        </div>

        <div className="field">
          <div className="row">
            <div>
              <label className="lbl" htmlFor="sid">
                Scene ID
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

        <div className="field">
          <button
            className="btn primary block"
            type="submit"
            disabled={busy || !source.trim()}
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
        <div className="field">
          <label className="lbl">Last ingest</label>
          <dl className="kv">
            <dt>Scene</dt>
            <dd>{report.scene_id}</dd>
            <dt>Indexed</dt>
            <dd>{report.tiles_indexed}</dd>
            <dt>Rejected</dt>
            <dd>{report.tiles_rejected}</dd>
            <dt>Seen</dt>
            <dd>{report.tiles_seen}</dd>
            <dt>Elapsed</dt>
            <dd>{report.elapsed_seconds}s</dd>
            <dt>Rate</dt>
            <dd>{report.tiles_per_second}/s</dd>
          </dl>
          {report.errors?.length > 0 && (
            <div className="notice error" style={{ margin: "10px 0 0" }}>
              {report.errors.length} error(s). First: {report.errors[0]}
            </div>
          )}
        </div>
      )}

      <div className="field">
        <label className="lbl">Indexed scenes</label>
        {scenes.length === 0 ? (
          <div className="hint">Nothing indexed yet.</div>
        ) : (
          <dl className="kv wide">
            {scenes.map((s) => (
              <div key={s.scene_id} style={{ display: "contents" }}>
                <dt>{s.scene_id}</dt>
                <dd>
                  {s.tile_count} tiles
                  {s.acquisition_timestamp
                    ? ` · ${s.acquisition_timestamp.slice(0, 10)}`
                    : ""}
                </dd>
              </div>
            ))}
          </dl>
        )}
        <button
          className="btn block"
          type="button"
          style={{ marginTop: 10 }}
          onClick={onRefresh}
          disabled={busy}
        >
          Refresh
        </button>
      </div>
    </div>
  );
}
