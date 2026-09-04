/**
 * Target detail + triage actions.
 *
 * The two verdict buttons write to the backend's append-only audit log. The
 * note field is sent with the verdict, so an analyst's reasoning is captured at
 * the moment of the decision rather than reconstructed later.
 */
import { useEffect, useState } from "react";
import { tilePreviewUrl } from "../api/client";

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
}) {
  const [note, setNote] = useState("");
  const [imgFailed, setImgFailed] = useState(false);

  // Reset per-tile state when the selection moves, or a note would leak across
  // targets and get attached to the wrong verdict.
  useEffect(() => {
    setNote(verdict?.analyst_note ?? "");
    setImgFailed(false);
  }, [hit?.tile_id, verdict?.analyst_note]);

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
