import logging
import time
import threading
import json
import queue
from pathlib import Path
from datetime import datetime

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from app.core.config import settings
from app.core.state import state, broadcaster

# Late imports inside the worker thread avoid blocking startup
logger = logging.getLogger("watchdog")

watchdog_queue = queue.Queue()

class DropZoneHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if path.name == "manifest.json":
            scene_dir = path.parent
            logger.info(f"Watchdog detected new scene drop: {scene_dir.name}")
            watchdog_queue.put(scene_dir)


def watchdog_worker():
    """Single worker thread processing scenes one by one."""
    while True:
        scene_dir = watchdog_queue.get()
        if scene_dir is None:
            break  # Poison pill
            
        try:
            _process_scene(scene_dir)
        except Exception as e:
            logger.error(f"Watchdog failed processing {scene_dir.name}: {e}")
        finally:
            watchdog_queue.task_done()


def _process_scene(scene_dir: Path):
    from app.api.routes import _GPU_LOCK
    from app.services.ingest import ingest_scene
    from app.services.vector_store import vector_store
    from app.services.change_detect import compare_scenes
    
    aoi = state.get_watchdog_aoi()
    if not aoi:
        logger.info("Watchdog processing skipped: No AOI armed.")
        return

    while state.should_pause_watchdog(30.0):
        logger.info("Watchdog paused due to recent /api/search activity.")
        time.sleep(5)
        
    logger.info(f"Watchdog initiating ML gate for {scene_dir.name}")

    with _GPU_LOCK:
        try:
            # 1. Ingest the scene into Qdrant for RemoteCLIP evaluation
            logger.info(f"Watchdog running ingest_scene on {scene_dir.name}...")
            ingest_scene(str(scene_dir), replace=True)
            
            # 2. RemoteCLIP Gate
            hits = vector_store.search(
                query=settings.watchdog_target_query,
                limit=10,
                score_threshold=settings.watchdog_similarity_threshold,
                scene_ids=[scene_dir.name],
                bounding_box=aoi
            )
            
            if not hits:
                logger.info(f"Watchdog RemoteCLIP Gate: No '{settings.watchdog_target_query}' targets found in {scene_dir.name}.")
                return
                
            logger.info(f"Watchdog RemoteCLIP Gate passed: Found {len(hits)} targets. Proceeding to Prithvi Gate.")
            
            # 3. Find Baseline for Prithvi (T1)
            scenes = vector_store.list_scenes()
            # Sort chronologically
            scenes.sort(key=lambda s: s["acquisition_timestamp"] or "")
            
            # We want the most recent scene BEFORE this new scene
            baseline_scene_id = None
            for s in reversed(scenes):
                if s["scene_id"] != scene_dir.name and s["scene_id"] < scene_dir.name:
                    baseline_scene_id = s["scene_id"]
                    break
                    
            if not baseline_scene_id:
                logger.warning("Watchdog Prithvi Gate skipped: No baseline scene available.")
                return
                
            # Assume baseline scene is in secure_drop_zone or sample_data
            baseline_path = settings.secure_drop_zone_dir / baseline_scene_id
            if not baseline_path.exists():
                baseline_path = settings.sample_data_dir / baseline_scene_id
            if not baseline_path.exists():
                logger.warning(f"Watchdog Prithvi Gate skipped: Baseline source {baseline_scene_id} missing on disk.")
                return

            # 4. Prithvi Gate: evaluate change for each identified target chip
            for hit in hits:
                logger.info(f"Watchdog evaluating change for row {hit['row']} col {hit['col']} against {baseline_scene_id}...")
                report = compare_scenes(
                    t1_source=str(baseline_path),
                    t2_source=str(scene_dir),
                    t1_scene_id=baseline_scene_id,
                    t2_scene_id=scene_dir.name,
                    row=hit["row"],
                    col=hit["col"],
                    min_change_score=settings.watchdog_change_threshold,
                    max_tiles=1,
                    skip_cloudy=True
                )
                
                # If any significant change was verified
                if report.results:
                    change_hit = report.results[0]
                    alert = {
                        "type": "target_detected",
                        "scene": scene_dir.name,
                        "tile_id": change_hit.tile_id,
                        "reason": f"Target identified by RemoteCLIP (score: {hit['score']:.2f}) & Prithvi change detected (score: {change_hit.change_score:.2f})",
                        "severity": "high"
                    }
                    broadcaster.broadcast(json.dumps(alert))
                    logger.warning(f"Watchdog ALERT broadcasted for {change_hit.tile_id}")
                else:
                    logger.info("Watchdog Prithvi Gate: Target found but no significant change detected.")

        except Exception as e:
            logger.error(f"Error during ML gate processing: {e}")


def start_watchdog():
    settings.secure_drop_zone_dir.mkdir(parents=True, exist_ok=True)

    # Start the worker thread
    worker = threading.Thread(target=watchdog_worker, name="watchdog_worker", daemon=True)
    worker.start()

    observer = PollingObserver(timeout=5.0)
    handler = DropZoneHandler()
    observer.schedule(handler, str(settings.secure_drop_zone_dir), recursive=True)
    observer.start()
    logger.info("Watchdog polling %s every 5s", settings.secure_drop_zone_dir)
    return observer

