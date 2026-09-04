/**
 * One retrieval or change hit. Renders as a <button> so keyboard users get
 * activation and focus handling for free.
 */
import { useState } from "react";
import { tilePreviewUrl } from "../api/client";

function Thumb({ tileId, hasPreview }) {
  const [failed, setFailed] = useState(false);

  if (!hasPreview || failed) {
    return (
      <div className="thumb missing" aria-hidden="true">
        NO PNG
      </div>
    );
  }
  return (
    <img
      className="thumb"
      src={tilePreviewUrl(tileId)}
      alt=""
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}

export default function ResultCard({ hit, mode, active, verdict, onSelect }) {
  const isChange = mode === "change";
  const score = isChange ? hit.change_score : hit.score;

  // RemoteCLIP cosine scores land around 0.15-0.35, and Prithvi change scores
  // are smaller still. A raw fraction-of-1.0 bar would be invisible, so scale
  // to a plausible ceiling per mode purely for the visual.
  const ceiling = isChange ? 0.15 : 0.4;
  const fill = Math.max(2, Math.min(100, ((score ?? 0) / ceiling) * 100));

  const cloud = hit.scl_cloud_coverage;
  const stamp = hit.acquisition_timestamp?.slice(0, 10);
  // Two scenes of the same ground produce hits with identical r,c. Leading with
  // the grid cell and putting the scene on the sub-line keeps them tellable
  // apart - the full tile_id is too long for the card and gets ellipsised.
  const cell = `r${hit.row ?? "?"}c${hit.col ?? "?"}`;

  return (
    <button
      type="button"
      className={`card ${verdict ?? ""}`}
      aria-current={active}
      onClick={() => onSelect(hit.tile_id)}
    >
      <Thumb tileId={hit.tile_id} hasPreview={Boolean(hit.preview_url) || isChange} />
      <div className="card-body">
        <div className="card-title">
          <span className="tile-id" title={hit.tile_id}>
            {cell}
          </span>
          <span className={`score ${isChange ? "change" : ""}`}>
            {score?.toFixed(4) ?? "—"}
          </span>
        </div>
        <div className="card-sub" title={hit.scene_id ?? hit.tile_id}>
          {stamp ?? hit.scene_id ?? "—"}
          {cloud != null ? ` · cloud ${(cloud * 100).toFixed(0)}%` : ""}
        </div>
        {verdict && (
          <div style={{ marginTop: 4 }}>
            <span className={`badge ${verdict}`}>
              {verdict === "confirmed" ? "confirmed" : "false alarm"}
            </span>
          </div>
        )}
        <div className={`bar ${isChange ? "change" : ""}`}>
          <i style={{ width: `${fill}%` }} />
        </div>
      </div>
    </button>
  );
}
