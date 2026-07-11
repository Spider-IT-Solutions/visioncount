import threading
import time
from collections import deque

import cv2

from pipeline.vision_pipeline import VisionPipeline
from utils.system_stats import get_system_stats


class ProcessingThread(threading.Thread):
    """Studio-only wrapper around VisionPipeline: camera I/O, frame-skip,
    JPEG encoding/caching into FrameBus, and manual recording. All actual
    detect/track/count logic lives in pipeline/vision_pipeline.py, shared
    with the generated per-project runtime.py."""

    def __init__(self, camera, frame_bus, cfg_manager, recorder):
        super().__init__(daemon=True)
        self.camera = camera
        self.frame_bus = frame_bus
        self.cfg_manager = cfg_manager
        self.recorder = recorder
        self.pipeline = VisionPipeline()

        self._running = True
        self._last_camera_cfg = None
        self._fps_window = deque(maxlen=30)
        self._frame_counter = 0
        self._last_annotated = None
        self._last_tracks = []
        self._last_stats_partial = {}
        self._last_debug = {}
        self._crossing_times = deque()

    def stop(self):
        self._running = False

    def reset_counts(self):
        self.pipeline.reset_counts()

    def reset_all(self):
        self.pipeline.reset_all()
        self._crossing_times.clear()
        self._last_annotated = None
        self._last_tracks = []
        self._last_stats_partial = {}
        self._last_debug = {}

    def run(self):
        while self._running:
            loop_start = time.time()
            cfg = self.cfg_manager.get_all()

            camera_cfg = cfg["camera"]
            if camera_cfg != self._last_camera_cfg:
                self.camera.reconfigure(camera_cfg)
                self._last_camera_cfg = camera_cfg

            frame = self.camera.get_frame()
            if frame is None:
                self.frame_bus.set_stats(self._build_stats(fps=0, proc_ms=0, tracks=[], stats_partial={}))
                time.sleep(0.02)
                continue

            frame_skip = max(0, int(cfg["performance"].get("frame_skip", 0)))
            self._frame_counter += 1
            skip_this_frame = frame_skip > 0 and (self._frame_counter % (frame_skip + 1) != 0)

            if skip_this_frame and self._last_annotated is not None:
                annotated = self._last_annotated
                tracks = self._last_tracks
                stats_partial = self._last_stats_partial
                debug = {}
                proc_ms = 0.0
            else:
                t0 = time.time()
                annotated, tracks, events, stats_partial, debug = self.pipeline.process(frame, cfg)
                proc_ms = (time.time() - t0) * 1000
                now = time.time()
                for _ in events:
                    self._crossing_times.append(now)
                while self._crossing_times and now - self._crossing_times[0] > 60:
                    self._crossing_times.popleft()
                self._last_annotated = annotated
                self._last_tracks = tracks
                self._last_stats_partial = stats_partial
                self._last_debug = debug

            annotated = self._draw_camera_status(annotated.copy())
            self.frame_bus.set_last_frame(annotated)

            if self.recorder.is_recording:
                self.recorder.write(annotated)

            self._publish_jpeg("main", annotated)
            active_debug = debug if debug else self._last_debug
            if "mask" in active_debug:
                self._publish_jpeg("mask", cv2.cvtColor(active_debug["mask"], cv2.COLOR_GRAY2BGR))
            if "hsv_result" in active_debug:
                self._publish_jpeg("hsv", active_debug["hsv_result"])

            loop_dt = time.time() - loop_start
            fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            self._fps_window.append(fps)

            stats = self._build_stats(
                fps=sum(self._fps_window) / len(self._fps_window), proc_ms=proc_ms,
                tracks=tracks, stats_partial=stats_partial,
            )
            self.frame_bus.set_stats(stats)

    def _publish_jpeg(self, key, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.frame_bus.set_jpeg(key, buf.tobytes() if ok else None)

    def _draw_camera_status(self, frame):
        if self.camera.ended:
            status, color = "ENDED", (0, 165, 255)
        elif self.camera.is_paused:
            status, color = "PAUSED", (0, 165, 255)
        elif self.camera.connected:
            status, color = "CONNECTED", (0, 200, 0)
        else:
            status, color = "DISCONNECTED", (0, 0, 255)
        cv2.putText(frame, status, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def _build_stats(self, fps, proc_ms, tracks, stats_partial):
        return {
            "fps": round(fps, 1),
            "processing_ms": round(proc_ms, 1),
            "camera_connected": self.camera.connected,
            "camera_ended": self.camera.ended,
            "camera_paused": self.camera.is_paused,
            "camera_is_file": self.camera.is_file_source,
            "camera_frame_index": self.camera.frame_index,
            "camera_frame_count": self.camera.frame_count,
            "object_count_current": stats_partial.get("object_count_current", len(tracks)),
            "counts": stats_partial.get("counts", {"per_line": {}, "total_in": 0, "total_out": 0, "total": 0}),
            "avg_confidence": stats_partial.get("avg_confidence", 0.0),
            "objects_per_minute": len(self._crossing_times),
            "recording": self.recorder.is_recording,
            "system": get_system_stats(),
        }
