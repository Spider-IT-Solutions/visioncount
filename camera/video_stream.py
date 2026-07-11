import os
import threading
import time

import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VideoStream(threading.Thread):
    """Threaded capture for file / USB / RTSP sources. Keeps only the latest
    frame (no queue backlog, so the dashboard never lags behind stale frames),
    auto-reconnects on read failure, and loops video files. reconfigure() lets
    the processing thread swap sources live when the operator changes camera
    settings — no restart.
    """

    def __init__(self, camera_cfg):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._cfg_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._frame = None
        self._connected = False
        self._ended = False
        self._paused = False
        self._pending_seek = None
        self._step_delta = None
        self._running = True
        self._cap = None
        self._current_cfg = None
        self._pending_cfg = dict(camera_cfg)

    def reconfigure(self, camera_cfg):
        with self._cfg_lock:
            self._pending_cfg = dict(camera_cfg)

    def restart(self):
        """Re-play a file source from frame 0. No-op for live sources."""
        with self._cap_lock:
            if self._cap is not None and self._current_cfg and self._current_cfg["source_type"] == "file":
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._ended = False
                self._paused = False

    def pause(self):
        if self.is_file_source:
            self._paused = True

    def resume(self):
        self._paused = False

    def step(self, delta=1):
        """Advance/rewind one frame (for-frame calibration) and stay paused
        on the landed frame. File sources only."""
        if self.is_file_source:
            self._step_delta = delta
            self._paused = True

    def seek(self, frame_number):
        """Jump to an absolute frame index (scrub bar) and pause there."""
        if self.is_file_source:
            self._pending_seek = max(0, int(frame_number))
            self._paused = True

    def _resolve_source(self, cfg):
        if cfg["source_type"] == "usb":
            return int(cfg["source"])
        if cfg["source_type"] == "file" and not os.path.isabs(str(cfg["source"])):
            return os.path.join(BASE_DIR, cfg["source"])
        return cfg["source"]

    def _open(self, cfg):
        cap = cv2.VideoCapture(self._resolve_source(cfg))
        if cfg["source_type"] in ("usb", "rtsp"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        return cap

    def run(self):
        with self._cfg_lock:
            self._current_cfg = dict(self._pending_cfg)
        self._cap = self._open(self._current_cfg)
        last_read_time = time.time()
        next_reconnect = 0.0

        while self._running:
            with self._cfg_lock:
                pending = dict(self._pending_cfg)
            if pending != self._current_cfg:
                with self._cap_lock:
                    if self._cap is not None:
                        self._cap.release()
                    self._current_cfg = pending
                    self._cap = self._open(self._current_cfg)
                    self._ended = False
                    self._paused = False
                    self._pending_seek = None
                    self._step_delta = None

            if self._cap is None or not self._cap.isOpened():
                self._connected = False
                now = time.time()
                if now >= next_reconnect:
                    with self._cap_lock:
                        if self._cap is not None:
                            self._cap.release()
                        self._cap = self._open(self._current_cfg)
                    next_reconnect = now + 2.0
                time.sleep(0.1)
                continue

            if self._ended:
                # Single-run file source finished — hold last frame, wait for
                # a loop-mode toggle, source change, or explicit restart().
                time.sleep(0.1)
                continue

            is_file = self._current_cfg["source_type"] == "file"
            landed_on_request = False
            if is_file and self._pending_seek is not None:
                target = self._pending_seek
                self._pending_seek = None
                self._step_delta = None
                with self._cap_lock:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                landed_on_request = True
            elif is_file and self._step_delta is not None:
                delta = self._step_delta
                self._step_delta = None
                if delta < 0:
                    with self._cap_lock:
                        cur = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur + delta - 1))
                landed_on_request = True
            elif is_file and self._paused:
                time.sleep(0.05)
                continue

            with self._cap_lock:
                ret, frame = self._cap.read()
            if not ret:
                if self._current_cfg["source_type"] == "file":
                    if self._current_cfg.get("loop", True):
                        with self._cap_lock:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._ended = True
                    continue
                self._connected = False
                self._cap.release()
                self._cap = None
                time.sleep(0.5)
                continue

            self._connected = True
            frame = cv2.resize(frame, (self._current_cfg["width"], self._current_cfg["height"]))
            with self._lock:
                self._frame = frame

            now = time.time()
            elapsed = now - last_read_time
            last_read_time = now
            if is_file and not landed_on_request:
                target = 1.0 / max(1, self._current_cfg["fps"])
                if elapsed < target:
                    time.sleep(target - elapsed)

        if self._cap is not None:
            self._cap.release()

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def connected(self):
        return self._connected

    @property
    def ended(self):
        return self._ended

    @property
    def is_paused(self):
        return self._paused

    @property
    def is_file_source(self):
        return self._current_cfg is not None and self._current_cfg["source_type"] == "file"

    @property
    def frame_index(self):
        if not self.is_file_source or self._cap is None:
            return None
        with self._cap_lock:
            pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, pos - 1)

    @property
    def frame_count(self):
        if not self.is_file_source or self._cap is None:
            return None
        with self._cap_lock:
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count or None

    def stop(self):
        self._running = False
