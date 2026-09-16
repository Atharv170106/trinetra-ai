import { useEffect } from "react";
import { useMap } from "./MapContext";
import "@geoman-io/leaflet-geoman-free";
import "@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css";

export default function MapDrawingTools({ onBoundingBoxChange }) {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    map.pm.addControls({
      position: "topright",
      drawMarker: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawPolygon: false,
      drawRectangle: true,
      drawCircle: false,
      drawText: false,
      cutPolygon: false,
    });

    const handleCreate = (e) => {
      const bounds = e.layer.getBounds();
      const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
      const roundedBbox = bbox.map((b) => parseFloat(b.toFixed(4)));
      if (onBoundingBoxChange) onBoundingBoxChange(roundedBbox);
    };

    const handleRemove = (e) => {
      if (onBoundingBoxChange) onBoundingBoxChange(null);
    };

    map.on("pm:create", handleCreate);
    map.on("pm:remove", handleRemove);

    return () => {
      map.pm.removeControls();
      map.off("pm:create", handleCreate);
      map.off("pm:remove", handleRemove);
    };
  }, [map, onBoundingBoxChange]);

  return null;
}
