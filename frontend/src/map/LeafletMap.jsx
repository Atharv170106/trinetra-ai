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
import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "@geoman-io/leaflet-geoman-free";
import "@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css";

import { graticuleLayer } from "./GraticuleLayer";
import { tilePreviewUrl } from "../api/client";

const CYAN = "#38bdf8";
const AMBER = "#f5a524";
const GREEN = "#34d399";
const RED = "#f87171";

// India-centred default view: the operational area for this deployment.
const DEFAULT_CENTER = [22.5, 79.0];
const DEFAULT_ZOOM = 5;

function strokeFor(hit, mode, verdict) {
  if (verdict === "confirmed") return GREEN;
  if (verdict === "false_alarm") return RED;
  return mode === "change" ? AMBER : CYAN;
}

export default function LeafletMap({
  hits,
  mode,
  selectedId,
  verdicts,
  showImagery,
  onSelect,
  onBoundingBoxChange,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const overlaysRef = useRef(null);
  const boxesRef = useRef(null);
  const shapesRef = useRef(new Map());
  // Refit only when the result set genuinely changes, not when a selection or a
  // verdict does - otherwise clicking a card would yank the viewport around.
  const fitKeyRef = useRef("");

  // ---------------------------------------------------------------- init once
  useEffect(() => {
    const map = L.map(containerRef.current, {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: true,
      attributionControl: true,
      preferCanvas: true,
      // Chip overlays are small and numerous; fading them on every pan is
      // visually noisy and costs frames on an 8 GB laptop.
      fadeAnimation: false,
    });

    graticuleLayer().addTo(map);
    overlaysRef.current = L.layerGroup().addTo(map);
    boxesRef.current = L.layerGroup().addTo(map);

    map.attributionControl.setPrefix(false);
    map.attributionControl.addAttribution(
      "Trinetra AI — offline graticule, no external tiles"
    );

    const legend = L.control({ position: "bottomleft" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "map-legend");
      div.innerHTML = `
        <div><span class="swatch" style="background:${CYAN}"></span>Retrieval hit</div>
        <div><span class="swatch" style="background:${AMBER}"></span>Change detected</div>
        <div><span class="swatch" style="background:${GREEN}"></span>Confirmed intel</div>
        <div><span class="swatch" style="background:${RED}"></span>False alarm</div>`;
      return div;
    };
    legend.addTo(map);

    const scale = L.control.scale({ imperial: false, position: "bottomright" });
    scale.addTo(map);

    map.pm.addControls({
      position: 'topright',
      drawMarker: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawPolygon: false,
      drawRectangle: true,
      drawCircle: false,
      drawText: false,
      cutPolygon: false,
    });

    map.on('pm:create', (e) => {
      const bounds = e.layer.getBounds();
      const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
      // Format to max 4 decimal places for cleanliness
      const roundedBbox = bbox.map(b => parseFloat(b.toFixed(4)));
      if (onBoundingBoxChange) onBoundingBoxChange(roundedBbox);
    });

    map.on('pm:remove', (e) => {
      // Clear bounding box if the drawn layer is removed
      // Assuming only one layer is drawn for simplicity.
      if (onBoundingBoxChange) onBoundingBoxChange(null);
    });

    mapRef.current = map;
    // Leaflet mis-measures its container if the parent grid settles after mount.
    const raf = requestAnimationFrame(() => map.invalidateSize());

    return () => {
      cancelAnimationFrame(raf);
      map.remove();
      mapRef.current = null;
      shapesRef.current.clear();
    };
  }, []);

  // --------------------------------------------------- redraw on result change
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    overlaysRef.current.clearLayers();
    boxesRef.current.clearLayers();
    shapesRef.current.clear();

    const bounds = [];

    hits.forEach((hit) => {
      const bbox = hit.wgs84_bounding_box;
      if (!bbox || bbox.length !== 4) return;
      const [w, s, e, n] = bbox;
      const rect = [
        [s, w],
        [n, e],
      ];
      bounds.push(rect);

      if (showImagery && hit.preview_url) {
        L.imageOverlay(tilePreviewUrl(hit.tile_id), rect, {
          opacity: 0.9,
          interactive: false,
          // A failed PNG must not leave a broken-image glyph on the map.
          errorOverlayUrl:
            "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        }).addTo(overlaysRef.current);
      }

      const verdict = verdicts?.[hit.tile_id]?.verdict;
      const box = L.rectangle(rect, {
        color: strokeFor(hit, mode, verdict),
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: showImagery ? 0 : 0.12,
      });

      const score = mode === "change" ? hit.change_score : hit.score;
      box.bindTooltip(
        `<b>${hit.tile_id}</b><br/>${mode === "change" ? "change" : "score"}: ${
          score?.toFixed(4) ?? "—"
        }`,
        { direction: "top", opacity: 0.92 }
      );
      box.on("click", () => onSelect(hit.tile_id));
      box.addTo(boxesRef.current);
      shapesRef.current.set(hit.tile_id, box);
    });

    // Fit only on a genuinely new result set.
    const fitKey = hits.map((h) => h.tile_id).join("|");
    if (bounds.length && fitKey !== fitKeyRef.current) {
      map.fitBounds(L.latLngBounds(bounds.flat()), {
        padding: [40, 40],
        maxZoom: 14,
      });
      fitKeyRef.current = fitKey;
    }
    if (!hits.length) fitKeyRef.current = "";
  }, [hits, mode, verdicts, showImagery, onSelect]);

  // ------------------------------------------------------ highlight selection
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    shapesRef.current.forEach((shape, tileId) => {
      const active = tileId === selectedId;
      shape.setStyle({
        weight: active ? 3 : 1.5,
        // Restyling alone can leave the active box beneath its neighbours.
        opacity: active ? 1 : 0.9,
      });
      if (active) shape.bringToFront();
    });

    if (selectedId && shapesRef.current.has(selectedId)) {
      const target = shapesRef.current.get(selectedId).getBounds();
      // Pan into view only if the chip is off-screen; never re-zoom under the
      // analyst while they are working a target.
      if (!map.getBounds().contains(target)) {
        map.panTo(target.getCenter(), { animate: true, duration: 0.35 });
      }
    }
  }, [selectedId, hits]);

  return (
    <>
      <div ref={containerRef} role="application" aria-label="Imagery map" />
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
