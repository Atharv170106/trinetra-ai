import React from "react";

export default function TacticalFilters({
  show,
  onClose,
  scenes,
  startDate,
  setStartDate,
  endDate,
  setEndDate,
  sceneId,
  setSceneId,
  limit,
  setLimit,
  threshold,
  setThreshold,
  maxCloud,
  setMaxCloud,
  onApply,
  onReset
}) {
  if (!show) return null;

  return (
    <div className="filters-popover tactical-reveal">
      <div className="filters-head">
        <span>TACTICAL SENSOR FILTERS</span>
        <button type="button" className="popover-close-btn" onClick={onClose}>
          ×
        </button>
      </div>

      <div className="filter-row">
        <div className="filter-col">
          <label>START DATE</label>
          <input type="date" className="filter-field" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
        <div className="filter-col">
          <label>END DATE</label>
          <input type="date" className="filter-field" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        </div>
      </div>

      <div className="filter-row">
        <div className="filter-col">
          <label>RESTRICT SCENE</label>
          <select className="filter-field" value={sceneId} onChange={(e) => setSceneId(e.target.value)}>
            <option value="">All Ingested MGRS Tiles</option>
            {scenes.map((s) => (
              <option key={s.scene_id} value={s.scene_id}>
                {s.scene_id} ({s.tile_count} chips)
              </option>
            ))}
          </select>
        </div>
        <div className="filter-col">
          <label>TOP K RESULTS</label>
          <select className="filter-field" value={limit} onChange={(e) => setLimit(e.target.value)}>
            <option value="10">10 chips</option>
            <option value="20">20 chips</option>
            <option value="50">50 chips</option>
            <option value="100">100 chips</option>
          </select>
        </div>
      </div>

      <div className="filter-slider-group">
        <div className="slider-header">
          <label>MIN COSINE SIMILARITY</label>
          <span>{threshold === 0 ? "AUTO" : threshold.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          className="tactical-slider"
          min="0" 
          max="0.4" 
          step="0.01" 
          value={threshold} 
          onChange={(e) => setThreshold(Number(e.target.value))} 
        />
      </div>

      <div className="filter-slider-group">
        <div className="slider-header">
          <label>MAX CLOUD PERMITTED (SCL)</label>
          <span>{maxCloud === 100 ? "UNCONSTRAINED" : `${maxCloud}%`}</span>
        </div>
        <input 
          type="range" 
          className="tactical-slider"
          min="0" 
          max="100" 
          step="5" 
          value={maxCloud} 
          onChange={(e) => setMaxCloud(Number(e.target.value))} 
        />
      </div>

      <div className="popover-footer">
        <button type="button" className="btn text-btn" onClick={onReset}>
          Reset Filters
        </button>
        <button type="button" className="btn primary tactical-glow-btn" onClick={onApply}>
          Apply & Scan
        </button>
      </div>
    </div>
  );
}
