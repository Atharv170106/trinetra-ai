import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Air-gap: every dependency resolves from node_modules. Leaflet's CSS is
// imported from the package, never a CDN link tag.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // Dev only. In Phase 6 the built assets are served from the same origin as
    // the API, so this proxy disappears and no CORS is involved at all.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    // Vite inlines small assets as data URIs; keep that so the built bundle
    // makes zero extra requests in an offline container.
    assetsInlineLimit: 8192,
  },
});
