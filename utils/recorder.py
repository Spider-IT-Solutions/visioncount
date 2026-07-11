import datetime
import os
import threading

import cv2

DEFAULT_RECORDING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings"
)


class Recorder:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or DEFAULT_RECORDING_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._writer = None
        self._filename = None

    @property
    def is_recording(self):
        with self._lock:
            return self._writer is not None

    def start(self, width, height, fps):
        with self._lock:
            if self._writer is not None:
                return self._filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._filename = f"rec_{timestamp}.mp4"
            path = os.path.join(self.base_dir, self._filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, max(1, fps), (width, height))
            return self._filename

    def write(self, frame):
        with self._lock:
            if self._writer is not None:
                self._writer.write(frame)

    def stop(self):
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            filename = self._filename
            self._filename = None
            return filename
