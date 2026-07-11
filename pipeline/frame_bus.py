import threading


class FrameBus:
    """Holds only the latest values (frame/jpegs/stats) behind small locks —
    deliberately not a queue, so viewers always see the freshest frame instead
    of draining a backlog and falling behind (keeps stream latency low)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpegs = {"main": None, "mask": None, "hsv": None}
        self._last_frame = None
        self._stats = {}

    def set_jpeg(self, key, data):
        with self._lock:
            self._jpegs[key] = data

    def get_jpeg(self, key):
        with self._lock:
            return self._jpegs.get(key)

    def set_last_frame(self, frame):
        with self._lock:
            self._last_frame = frame

    def get_last_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def set_stats(self, stats):
        with self._lock:
            self._stats = stats

    def get_stats(self):
        with self._lock:
            return dict(self._stats)


frame_bus = FrameBus()
