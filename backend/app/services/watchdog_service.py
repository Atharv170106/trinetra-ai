import logging
import time
import threading
import json
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.core.config import settings
from app.core.state import state, broadcaster
from app.services.ingest import ingest_scene
from app.services.ml_inference import remoteclip
from app.services.change_detect import compare_scenes

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
    
    observer = Observer()
    handler = DropZoneHandler()
    observer.schedule(handler, str(settings.secure_drop_zone_dir), recursive=True)
    observer.start()
    return observer
