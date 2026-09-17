/**
 * Target detail + triage actions.
 *
 * The two verdict buttons write to the backend's append-only audit log. The
 * note field is sent with the verdict, so an analyst's reasoning is captured at
 * the moment of the decision rather than reconstructed later.
 */
import { useEffect, useState } from "react";
import { api, tilePreviewUrl } from "../api/client";

function fmt(value, digits = 6) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

export default function MetadataDrawer({
  hit,
  mode,
  verdict,
  busy,
  onTriage,
  onClose,
  query = null,
  baselineSceneId = null,
}) {
  const [note, setNote] = useState("");
  const [imgFailed, setImgFailed] = useState(false);

  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiResult, setXaiResult] = useState("");
  const [xaiError, setXaiError] = useState("");

  // Reset per-tile state when the selection moves, or a note would leak across
  // targets and get attached to the wrong verdict.
  useEffect(() => {
    setNote(verdict?.analyst_note ?? "");
    setImgFailed(false);
    setXaiResult("");
    setXaiError("");
  }, [hit?.tile_id, verdict?.analyst_note]);

  // In change mode the drawer holds the T2 chip. The matching T1 chip has the
  // same r/c suffix under the baseline scene, so it can be derived rather than
  // plumbed through - the change response carries no baseline_tile_id field.
  const baselineTileId = (() => {
    if (mode !== "change" || !baselineSceneId || !hit) return null;
    const suffix = String(hit.tile_id ?? "").match(/_r\d{4}c\d{4}$/)?.[0];
    if (suffix) return `${baselineSceneId}${suffix}`;
    if (typeof hit.row === "number" && typeof hit.col === "number") {
      const p = (n) => String(n).padStart(4, "0");
      return `${baselineSceneId}_r${p(hit.row)}c${p(hit.col)}`;
    }
    return null;
  })();

  const handleExplain = async () => {
    setXaiLoading(true);
    setXaiResult("");
    setXaiError("");
    try {
      const res = await api.explain({
        query: query || "Assess this satellite imagery chip for items of military interest.",
        targetTileId: hit.tile_id,
        baselineTileId,
      });
      setXaiResult(res.summary);
    } catch (err) {
      setXaiError(err.message);
    } finally {
      setXaiLoading(false);
    }
  };

  if (!hit) return null;

  const bbox = hit.wgs84_bounding_box ?? [];
  const isChange = mode === "change";


  return (
    <aside className="drawer" aria-label="Target details">
      <div className="drawer-head">
        <h2>Target detail</h2>
        <button className="icon-btn" onClick={onClose} aria-label="Close details">
          ×
        </button>
      </div>

      <div className="drawer-body">
        {imgFailed ? (
          <div className="notice">
            No cached preview for this chip. Change-analysis results are computed
            on demand and are not written to the tile cache.
          </div>
        ) : (
          <img
            className="preview-full"
            src={tilePreviewUrl(hit.tile_id)}
            alt={`Imagery chip ${hit.tile_id}`}
            onError={() => setImgFailed(true)}
          />
        )}

        <dl className="kv">
          <dt>Tile</dt>
          <dd>{hit.tile_id}</dd>

          <dt>Scene</dt>
          <dd>{hit.scene_id ?? "—"}</dd>

          <dt>Grid</dt>
          <dd>
            r{hit.row ?? "?"} c{hit.col ?? "?"}
          </dd>

          {isChange ? (
            <>
              <dt>Change</dt>
              <dd style={{ color: "var(--amber)" }}>{fmt(hit.change_score)}</dd>
              <dt>Cosine dist</dt>
              <dd>{fmt(hit.cosine_distance)}</dd>
              <dt>L2 dist</dt>
              <dd>{fmt(hit.l2_distance, 4)}</dd>
              <dt>Cloud T1/T2</dt>
              <dd>
                {fmt(hit.t1_cloud, 3)} / {fmt(hit.t2_cloud, 3)}
              </dd>
            </>
          ) : (
            <>
              <dt>Similarity</dt>
              <dd style={{ color: "var(--cyan)" }}>{fmt(hit.score)}</dd>
              <dt>Acquired</dt>
              <dd>{hit.acquisition_timestamp ?? "—"}</dd>
              <dt>Cloud</dt>
              <dd>
                {hit.scl_cloud_coverage != null
                  ? `${(hit.scl_cloud_coverage * 100).toFixed(1)}%`
                  : "—"}
              </dd>
            </>
          )}

          <dt>W, S</dt>
          <dd>
            {fmt(bbox[0], 5)}, {fmt(bbox[1], 5)}
          </dd>
          <dt>E, N</dt>
          <dd>
            {fmt(bbox[2], 5)}, {fmt(bbox[3], 5)}
          </dd>

          {verdict && (
            <>
              <dt>Verdict</dt>
              <dd>
                <span className={`badge ${verdict.verdict}`}>
                  {verdict.verdict === "confirmed" ? "confirmed" : "false alarm"}
                </span>
              </dd>
              <dt>Logged</dt>
              <dd>{verdict.logged_at?.slice(0, 19).replace("T", " ")}</dd>
            </>
          )}
        </dl>
      </div>

      <div className="drawer-foot">
        {xaiError && (
          <div className="notice error" style={{ marginBottom: "1rem" }}>
            ❌ {xaiError}
          </div>
        )}
        
        {xaiResult && (
          <div style={{ marginBottom: "1rem", padding: "0.75rem", background: "var(--bg-lighter)", border: "1px solid var(--border)", borderRadius: "4px" }}>
            <h4 style={{ margin: "0 0 0.5rem", fontSize: "0.75rem", textTransform: "uppercase", color: "var(--accent)" }}>🧠 Explainable AI Summary</h4>
            <div style={{ fontSize: "0.85rem", lineHeight: 1.4, whiteSpace: "pre-wrap" }}>
              {xaiResult}
            </div>
            <div style={{ marginTop: "0.5rem", fontSize: "0.65rem", opacity: 0.55 }}>
              {baselineTileId
                ? "Local VLM, T1 vs T2 comparison — unverified, corroborate before acting."
                : "Local VLM, single chip — unverified, corroborate before acting."}
            </div>
          </div>
        )}

        <div style={{ marginBottom: "1rem" }}>
          <button
            className={`btn block ${xaiLoading ? "busy-pulse" : ""}`}
            style={{ width: "100%", background: "var(--bg-lighter)", border: "1px solid var(--accent)", color: "var(--accent)" }}
            onClick={handleExplain}
            disabled={busy || xaiLoading}
            title={baselineTileId
              ? `Compares ${baselineTileId} against ${hit.tile_id} on the local GPU`
              : "Runs a local vision model on this chip — briefly uses GPU VRAM"}
          >
            {xaiLoading
              ? "⏳ Generating Intel Summary..."
              : baselineTileId
                ? "🤖 Explain Change T1 → T2 (XAI)"
                : "🤖 Generate Intel Summary (XAI)"}
          </button>
        </div>

        <div>
          <label className="lbl" htmlFor="note">
            Analyst note
          </label>
          <textarea
            id="note"
            className="txt"
            rows={2}
            placeholder="Optional — recorded with the verdict"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>
        <div className="row">
          <button
            className="btn confirm"
            disabled={busy}
            onClick={() => onTriage(hit.tile_id, "confirmed", note)}
          >
            ✓ Confirm intel
          </button>
          <button
            className="btn reject"
            disabled={busy}
            onClick={() => onTriage(hit.tile_id, "false_alarm", note)}
          >
            ✗ False alarm
          </button>
        </div>
      </div>
    </aside>
  );
}
