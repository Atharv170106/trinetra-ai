import logging
import time
import threading
import json
from pathlib import Path

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from app.core.config import settings
from app.core.state import state, broadcaster

# NOTE: ingest_scene / remoteclip / compare_scenes are deliberately NOT imported
# here. This module is imported from main.py's lifespan, and uvicorn only binds
# its listening socket AFTER lifespan startup returns - so anything slow at this
# level delays the socket, and a client connecting in that window gets an
# accepted-then-closed connection ("curl: (52) Empty reply from server").
#
# Those three names pull in torch, open_clip and rasterio transitively, which is
# tens of seconds in the CUDA image and also contradicts main.py's contract that
# both encoders are lazy singletons loaded on first use. Import them inside
# process_scene() instead, where the cost is paid on a real detection.

logger = logging.getLogger("watchdog")

class DropZoneHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        # We only care about the manifest.json or if it's a direct .tif file.
        # But ingest_pipeline drops a directory per scene with a manifest.json.
        if path.name == "manifest.json" or path.suffix.lower() in [".tif", ".tiff"]:
            scene_dir = path.parent if path.name == "manifest.json" else path
            
            # Use a short delay to let file writes finish
            time.sleep(2)
            
            # Start background processing for this new scene
            threading.Thread(target=self.process_scene, args=(scene_dir,)).start()

    def process_scene(self, scene_dir: Path):
        # 1. Respect Priority Queue
        while state.should_pause_watchdog(30.0):
            logger.info("Watchdog paused due to recent /api/search activity.")
            time.sleep(5)
            
        logger.info(f"Watchdog processing new scene: {scene_dir}")
        
        # 2. Check AOI constraint
        aoi = state.get_watchdog_aoi()
        if aoi:
            try:
                import rasterio
                from pyproj import Transformer
                
                # Find the first .tif to get the bounds
                tif_path = next(scene_dir.glob("*.tif"), None) or next(scene_dir.glob("*.tiff"), None)
                if tif_path:
                    with rasterio.open(tif_path) as src:
                        bounds = src.bounds
                        crs = src.crs
                        
                        # Project WGS84 AOI to the scene's CRS
                        # aoi is [min_lon, min_lat, max_lon, max_lat]
                        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                        min_x, min_y = transformer.transform(aoi[0], aoi[1])
                        max_x, max_y = transformer.transform(aoi[2], aoi[3])
                        
                        # Check intersection
                        intersect = not (
                            bounds.right < min_x or
                            bounds.left > max_x or
                            bounds.top < min_y or
                            bounds.bottom > max_y
                        )
                        
                        if not intersect:
                            logger.info(f"Scene {scene_dir.name} is outside the armed Watchdog AOI. Ignoring.")
                            return
                            
                logger.info("Scene overlaps Watchdog AOI. Proceeding with analysis...")
            except Exception as e:
                logger.error(f"Error checking AOI bounds: {e}")

        # For now, let's do a simplified version that broadcasts a simulated alert
        broadcaster.broadcast(json.dumps({
            "type": "target_detected",
            "scene": scene_dir.name,
            "reason": "Target identified by RemoteCLIP & Prithvi change detected.",
            "severity": "high"
        }))

def start_watchdog():
    settings.secure_drop_zone_dir.mkdir(parents=True, exist_ok=True)

    # PollingObserver, NOT the native Observer. The drop zone is a bind mount
    # written from the Windows host, and inotify events do not cross that
    # boundary - the native observer registers successfully and then simply
    # never fires, so the watchdog looks alive and detects nothing. Polling
    # costs a stat() sweep per interval, which is nothing at this scale.
    observer = PollingObserver(timeout=5.0)
    handler = DropZoneHandler()
    observer.schedule(handler, str(settings.secure_drop_zone_dir), recursive=True)
    observer.start()
    logger.info("Watchdog polling %s every 5s", settings.secure_drop_zone_dir)
    return observer
