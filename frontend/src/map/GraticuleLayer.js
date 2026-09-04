/**
 * Trinetra AI - offline coordinate graticule.
 *
 * There is no basemap. An air-gapped deployment has no tile server and no CDN,
 * so rather than ship a fake grey canvas we draw a real lat/lon graticule whose
 * spacing adapts to zoom. Analysts get positional context from the grid and from
 * the georeferenced chip overlays; nothing is fetched over the network.
 *
 * Implemented as a Leaflet GridLayer so panning/zooming reuses Leaflet's own
 * tile lifecycle instead of a full redraw on every move.
 */
import L from "leaflet";

// Degrees between graticule lines at each zoom level. Below z3 a 30-degree grid
// keeps the world readable; past z11 finer than 0.01 degrees is noise.
const SPACING = [30, 30, 30, 10, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01];

function spacingFor(zoom) {
  return SPACING[Math.max(0, Math.min(SPACING.length - 1, Math.round(zoom)))];
}

function label(value, axis) {
  const abs = Math.abs(value);
  const hemi = axis === "lat" ? (value >= 0 ? "N" : "S") : value >= 0 ? "E" : "W";
  const decimals = abs < 1 ? 3 : abs < 10 ? 2 : 1;
  return `${abs.toFixed(decimals)}°${hemi}`;
}

const Graticule = L.GridLayer.extend({
  options: {
    lineColor: "#233043",
    majorColor: "#31425b",
    textColor: "#5b6b81",
    font: "10px ui-monospace, monospace",
  },

  createTile(coords) {
    const tile = L.DomUtil.create("canvas", "leaflet-tile");
    const size = this.getTileSize();
    // Draw at device resolution so the 1px lines stay crisp on a HiDPI laptop.
    const dpr = window.devicePixelRatio || 1;
    tile.width = size.x * dpr;
    tile.height = size.y * dpr;
    tile.style.width = `${size.x}px`;
    tile.style.height = `${size.y}px`;

    const ctx = tile.getContext("2d");
    if (!ctx) return tile;
    ctx.scale(dpr, dpr);

    const map = this._map;
    if (!map) return tile;

    const step = spacingFor(coords.z);
    const nw = map.unproject(coords.scaleBy(size), coords.z);
    const se = map.unproject(coords.add([1, 1]).scaleBy(size), coords.z);

    const north = Math.max(nw.lat, se.lat);
    const south = Math.min(nw.lat, se.lat);
    const west = Math.min(nw.lng, se.lng);
    const east = Math.max(nw.lng, se.lng);

    const origin = coords.scaleBy(size);
    ctx.lineWidth = 1;
    ctx.font = this.options.font;
    ctx.fillStyle = this.options.textColor;
    ctx.textBaseline = "top";

    // Parallels
    const firstLat = Math.floor(south / step) * step;
    for (let lat = firstLat; lat <= north + step; lat += step) {
      if (lat < -85 || lat > 85) continue;
      const pt = map.project([lat, west], coords.z).subtract(origin);
      const y = Math.round(pt.y) + 0.5;
      if (y < -1 || y > size.y + 1) continue;
      // Lines on a whole multiple of 10x the step read as major.
      const major = Math.abs(lat % (step * 10)) < 1e-9;
      ctx.strokeStyle = major ? this.options.majorColor : this.options.lineColor;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size.x, y);
      ctx.stroke();
      if (major || step >= 1) ctx.fillText(label(lat, "lat"), 4, y + 3);
    }

    // Meridians
    const firstLng = Math.floor(west / step) * step;
    for (let lng = firstLng; lng <= east + step; lng += step) {
      const pt = map.project([south, lng], coords.z).subtract(origin);
      const x = Math.round(pt.x) + 0.5;
      if (x < -1 || x > size.x + 1) continue;
      const major = Math.abs(lng % (step * 10)) < 1e-9;
      ctx.strokeStyle = major ? this.options.majorColor : this.options.lineColor;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, size.y);
      ctx.stroke();
      if (major || step >= 1) {
        ctx.save();
        ctx.translate(x + 3, 4);
        ctx.fillText(label(lng, "lng"), 0, 0);
        ctx.restore();
      }
    }

    return tile;
  },
});

export function graticuleLayer(options) {
  return new Graticule(options);
}
