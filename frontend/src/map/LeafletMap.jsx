/**
 * Trinetra AI - map canvas.
 *
 * Leaflet is driven imperatively through refs rather than recreated on render:
 * a React-managed map instance would tear down and rebuild the whole canvas on
 * every state change, losing pan/zoom and thrashing the chip overlays.
 *
 * Two layers carry the results:
 *   overlays  - the cached tile PNGs, georeferenced onto their real bboxes.
 *               This is the imagery. There is no basemap in an air-gapped
 *               deployment, so the chips ARE the map.
 *   boxes     - rectangles for hit extent, colour-coded by score or verdict.
 */
import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import { graticuleLayer } from "./GraticuleLayer";
import { tilePreviewUrl } from "../api/client";
import { MapContext } from "./MapContext";
import MapDrawingTools from "./MapDrawingTools";

const CYAN = "#38bdf8";
const AMBER = "#f5a524";
const GREEN = "#34d399";
const RED = "#f87171";
const MAGENTA = "#d946ef"; // For top change hit

// India-centred default view: the operational area for this deployment.
const DEFAULT_CENTER = [22.5, 79.0];
const DEFAULT_ZOOM = 5;

function strokeFor(hit, mode, verdict, isTopHit) {
  if (verdict === "confirmed") return GREEN;
  if (verdict === "false_alarm") return RED;
  if (mode === "change") return isTopHit ? MAGENTA : AMBER;
  return CYAN;
}

export default function LeafletMap({
  hits,
  mode,
  selectedId,
  verdicts,
  showImagery,
  meta,
  onSelect,
  onBoundingBoxChange,
}) {
  const container1Ref = useRef(null);
  const container2Ref = useRef(null);
  const [map1Instance, setMap1Instance] = useState(null);
  
  const map1Ref = useRef(null);
  const map2Ref = useRef(null);
  
  const overlays1Ref = useRef(null);
  const overlays2Ref = useRef(null);
  const boxes1Ref = useRef(null);
  const boxes2Ref = useRef(null);
  
  const shapes1Ref = useRef(new Map());
  const shapes2Ref = useRef(new Map());

  // Refit only when the result set genuinely changes
  const fitKeyRef = useRef("");

  // ---------------------------------------------------------------- init once
  useEffect(() => {
    const config = {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: true,
      attributionControl: true,
      preferCanvas: true,
      fadeAnimation: false,
    };

    const map1 = L.map(container1Ref.current, config);
    const map2 = L.map(container2Ref.current, { ...config, zoomControl: false, attributionControl: false });

    graticuleLayer().addTo(map1);
    graticuleLayer().addTo(map2);

    overlays1Ref.current = L.layerGroup().addTo(map1);
    overlays2Ref.current = L.layerGroup().addTo(map2);
    boxes1Ref.current = L.layerGroup().addTo(map1);
    boxes2Ref.current = L.layerGroup().addTo(map2);

    map1.attributionControl.setPrefix(false);
    map1.attributionControl.addAttribution("Trinetra AI — offline graticule, no external tiles");

    const legend = L.control({ position: "bottomleft" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "map-legend");
      div.innerHTML = `
        <div><span class="swatch" style="background:${CYAN}"></span>Retrieval hit</div>
        <div><span class="swatch" style="background:${AMBER}"></span>Change detected</div>
        <div><span class="swatch" style="background:${MAGENTA}"></span>Top change hit</div>
        <div><span class="swatch" style="background:${GREEN}"></span>Confirmed intel</div>
        <div><span class="swatch" style="background:${RED}"></span>False alarm</div>`;
      return div;
    };
    legend.addTo(map1);

    const scale = L.control.scale({ imperial: false, position: "bottomright" });
    scale.addTo(map1);

    // Sync maps
    let isSyncing = false;
    map1.on('move', () => {
      if (!isSyncing) {
        isSyncing = true;
        map2.setView(map1.getCenter(), map1.getZoom(), { animate: false });
        isSyncing = false;
      }
    });
    map2.on('move', () => {
      if (!isSyncing) {
        isSyncing = true;
        map1.setView(map2.getCenter(), map2.getZoom(), { animate: false });
        isSyncing = false;
      }
    });

    map1Ref.current = map1;
    map2Ref.current = map2;
    setMap1Instance(map1);

    const raf = requestAnimationFrame(() => {
      map1.invalidateSize();
      map2.invalidateSize();
    });

    return () => {
      cancelAnimationFrame(raf);
      map1.remove();
      map2.remove();
      map1Ref.current = null;
      map2Ref.current = null;
      setMap1Instance(null);
      shapes1Ref.current.clear();
      shapes2Ref.current.clear();
    };
  }, []);

  // When mode changes, invalidate size so maps adjust to flex changes
  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (map1Ref.current) map1Ref.current.invalidateSize();
      if (map2Ref.current) map2Ref.current.invalidateSize();
    });
    return () => cancelAnimationFrame(raf);
  }, [mode]);

  // --------------------------------------------------- redraw on result change
  useEffect(() => {
    const map1 = map1Ref.current;
    const map2 = map2Ref.current;
    if (!map1 || !map2) return;

    overlays1Ref.current.clearLayers();
    overlays2Ref.current.clearLayers();
    boxes1Ref.current.clearLayers();
    boxes2Ref.current.clearLayers();
    shapes1Ref.current.clear();
    shapes2Ref.current.clear();

    const bounds = [];
    const isChange = mode === "change";

    hits.forEach((hit, index) => {
      const bbox = hit.wgs84_bounding_box;
      if (!bbox || bbox.length !== 4) return;
      const [w, s, e, n] = bbox;
      const rect = [[s, w], [n, e]];
      bounds.push(rect);

      const isTopHit = isChange && index === 0;

      if (showImagery) {
        if (isChange && meta?.t1_scene_id && meta?.t2_scene_id) {
          // Construct tile IDs for T1 and T2
          const rowStr = hit.row.toString().padStart(4, "0");
          const colStr = hit.col.toString().padStart(4, "0");
          const t1_tile_id = `${meta.t1_scene_id}_r${rowStr}c${colStr}`;
          const t2_tile_id = `${meta.t2_scene_id}_r${rowStr}c${colStr}`;

          L.imageOverlay(tilePreviewUrl(t1_tile_id), rect, {
            opacity: 0.9, interactive: false,
            errorOverlayUrl: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
          }).addTo(overlays1Ref.current);

          L.imageOverlay(tilePreviewUrl(t2_tile_id), rect, {
            opacity: 0.9, interactive: false,
            errorOverlayUrl: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
          }).addTo(overlays2Ref.current);
        } else {
          // Normal search mode
          L.imageOverlay(tilePreviewUrl(hit.tile_id), rect, {
            opacity: 0.9, interactive: false,
            errorOverlayUrl: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
          }).addTo(overlays1Ref.current);
        }
      }

      const verdict = verdicts?.[hit.tile_id]?.verdict;
      const color = strokeFor(hit, mode, verdict, isTopHit);
      
      const box1 = L.rectangle(rect, { color, weight: 1.5, opacity: 0.9, fillOpacity: showImagery ? 0 : 0.12 });
      const score = mode === "change" ? hit.change_score : hit.score;
      const tooltipHTML = `<b>${hit.tile_id}</b><br/>${mode === "change" ? "change" : "score"}: ${score?.toFixed(4) ?? "—"}`;
      
      box1.bindTooltip(tooltipHTML, { direction: "top", opacity: 0.92 });
      box1.on("click", () => onSelect(hit.tile_id));
      box1.addTo(boxes1Ref.current);
      shapes1Ref.current.set(hit.tile_id, box1);

      if (isChange) {
        const box2 = L.rectangle(rect, { color, weight: 1.5, opacity: 0.9, fillOpacity: showImagery ? 0 : 0.12 });
        box2.bindTooltip(tooltipHTML, { direction: "top", opacity: 0.92 });
        box2.on("click", () => onSelect(hit.tile_id));
        box2.addTo(boxes2Ref.current);
        shapes2Ref.current.set(hit.tile_id, box2);
      }
    });

    const fitKey = hits.map((h) => h.tile_id).join("|");
    if (bounds.length && fitKey !== fitKeyRef.current) {
      map1.fitBounds(L.latLngBounds(bounds.flat()), { padding: [40, 40], maxZoom: 14 });
      // Map 2 will sync automatically via move event
      fitKeyRef.current = fitKey;
    }
    if (!hits.length) fitKeyRef.current = "";
  }, [hits, mode, verdicts, showImagery, meta, onSelect]);

  // ------------------------------------------------------ highlight selection
  useEffect(() => {
    const map1 = map1Ref.current;
    if (!map1) return;

    shapes1Ref.current.forEach((shape, tileId) => {
      const active = tileId === selectedId;
      shape.setStyle({ weight: active ? 3 : 1.5, opacity: active ? 1 : 0.9 });
      if (active) shape.bringToFront();
    });
    
    shapes2Ref.current.forEach((shape, tileId) => {
      const active = tileId === selectedId;
      shape.setStyle({ weight: active ? 3 : 1.5, opacity: active ? 1 : 0.9 });
      if (active) shape.bringToFront();
    });

    if (selectedId && shapes1Ref.current.has(selectedId)) {
      const target = shapes1Ref.current.get(selectedId).getBounds();
      if (!map1.getBounds().contains(target)) {
        map1.panTo(target.getCenter(), { animate: true, duration: 0.35 });
      }
    }
  }, [selectedId, hits]);

  const isChange = mode === "change";

  return (
    <>
      <div style={{ display: "flex", width: "100%", height: "100%", position: "relative" }}>
        {/* Map 1 */}
        <div style={{ flex: 1, position: "relative", borderRight: isChange ? "2px solid var(--border)" : "none" }}>
          <div ref={container1Ref} role="application" aria-label="Imagery map 1" style={{ width: "100%", height: "100%" }} />
          {isChange && meta?.t1_scene_id && (
            <div style={{ position: "absolute", top: 10, right: 10, zIndex: 1000, background: "rgba(0,0,0,0.7)", padding: "4px 8px", borderRadius: "4px", color: "#fff", fontWeight: "bold" }}>
              T1: {meta.t1_scene_id}
            </div>
          )}
        </div>
        
        {/* Map 2 */}
        <div style={{ flex: isChange ? 1 : 0, display: isChange ? "block" : "none", position: "relative" }}>
          <div ref={container2Ref} role="application" aria-label="Imagery map 2" style={{ width: "100%", height: "100%" }} />
          {isChange && meta?.t2_scene_id && (
            <div style={{ position: "absolute", top: 10, right: 10, zIndex: 1000, background: "rgba(0,0,0,0.7)", padding: "4px 8px", borderRadius: "4px", color: "#fff", fontWeight: "bold" }}>
              T2: {meta.t2_scene_id}
            </div>
          )}
        </div>
      </div>
      
      {map1Instance && (
        <MapContext.Provider value={map1Instance}>
          <MapDrawingTools onBoundingBoxChange={onBoundingBoxChange} />
        </MapContext.Provider>
      )}

      {!hits.length && (
        <div className="map-empty">
          <div>
            No results plotted.
            <br />
            Run a search or a change comparison.
          </div>
        </div>
      )}
    </>
  );
}
