import threading
import time

class PriorityQueueState:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_search_time = 0.0
        self._watchdog_aoi: list[float] | None = None

    def mark_search(self):
        with self._lock:
            self._last_search_time = time.time()

    def should_pause_watchdog(self, cooldown_seconds: float = 30.0) -> bool:
        with self._lock:
            return (time.time() - self._last_search_time) < cooldown_seconds

    def set_watchdog_aoi(self, bbox: list[float] | None):
        with self._lock:
            self._watchdog_aoi = bbox

    def get_watchdog_aoi(self) -> list[float] | None:
        with self._lock:
            return self._watchdog_aoi

state = PriorityQueueState()
import asyncio

class EventBroadcaster:
    def __init__(self):
        self.queues: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Record the server's event loop. Called once from main.py's lifespan.

        Without this, broadcast() silently dropped EVERY alert. It called
        asyncio.get_running_loop(), which raises RuntimeError when invoked from a
        plain worker thread - and the watchdog is exactly that. The except branch
        swallowed it, so the SSE stream stayed empty forever while the logs
        happily reported that an alert had been sent.
        """
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self.queues:
            self.queues.remove(q)

    def broadcast(self, message: str):
        """Thread-safe: callable from the watchdog worker or from the loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._loop
        if loop is None or loop.is_closed():
            # Nothing to deliver into yet. Say so rather than pretending.
            print(f"[broadcaster] no event loop bound; dropped: {message[:120]}")
            return
        for q in list(self.queues):
            loop.call_soon_threadsafe(q.put_nowait, message)

broadcaster = EventBroadcaster()

