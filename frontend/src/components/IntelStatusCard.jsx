import React from "react";

export default function IntelStatusCard() {
  return (
    <div className="intel-status-card">
      <div className="card-tag">
        <span className="live-dot pulse"></span>
        TACTICAL POSTURE: AIR-GAPPED
      </div>
      <div className="intel-desc">
        Omni-Search accepts natural language queries, bounding boxes, or coordinate arrays.
      </div>
      <div className="aoi-helper">
        <span className="helper-title">Spatial Watchdog:</span>
        Draw a bounding box on the map using the right-hand drawing toolbar to establish an AOI watch zone.
      </div>
    </div>
  );
}
