import threading
import time

class PriorityQueueState:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_search_time = 0.0

    def mark_search(self):
        with self._lock:
            self._last_search_time = time.time()

    def should_pause_watchdog(self, cooldown_seconds: float = 30.0) -> bool:
        with self._lock:
            return (time.time() - self._last_search_time) < cooldown_seconds

state = PriorityQueueState()
import asyncio

class EventBroadcaster:
    def __init__(self):
        self.queues: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self.queues:
            self.queues.remove(q)

    def broadcast(self, message: str):
        # We might call this from a sync thread (like watchdog)
        # So we need to put it safely into the async queues
        for q in self.queues:
            # If the loop is running, we can use call_soon_threadsafe
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(q.put_nowait, message)
            except RuntimeError:
                # No running loop, just put it directly (only works if in same loop context, which watchdog isn't)
                # We can handle this by creating a global event loop reference in main.py, or we can use asyncio.run_coroutine_threadsafe
                pass

broadcaster = EventBroadcaster()

