"""
VisionCount — Conveyor Belt Object Detection, Tracking & Counting
=================================================================
Entry point.  Reads config/settings.yaml, wires all modules together, and
runs the main processing loop.

Keyboard controls (when the display window is focused):
  q / ESC  → quit
  r        → reset counter and tracker
  b        → rebuild background model
  s        → save a manual snapshot
  p        → pause / resume
  +/-      → adjust counting line position on the fly
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── local imports ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.camera.capture import CameraCapture
from src.detection.background_subtractor import BackgroundSubtractorDetector, Detection
from src.tracking.centroid_tracker import CentroidTracker
from src.counting.line_counter import LineCounter
from src.data_logging.data_logger import DataLogger
from src.ui.display import Display
from utils.fps_counter import FPSCounter


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "logs/system.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        ),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=handlers,
    )


def _build_detector(cfg: dict):
    det_cfg = cfg["detection"]
    method = det_cfg.get("method", "background_subtraction")
    if method == "yolo":
        from src.detection.yolo_detector import YOLODetector
        yolo_cfg = cfg.get("yolo", {})
        return YOLODetector(
            model_path=yolo_cfg.get("model_path", "models/yolov8n.pt"),
            confidence=yolo_cfg.get("confidence", 0.5),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
            classes=yolo_cfg.get("classes"),
            device=yolo_cfg.get("device", "cpu"),
        )
    if method == "yolo_world":
        from src.detection.yolo_detector import YOLODetector
        yw_cfg = cfg.get("yolo_world", {})
        return YOLODetector(
            model_path=yw_cfg.get("model_path", "models/yolov8s-worldv2.pt"),
            confidence=yw_cfg.get("confidence", 0.05),
            iou_threshold=yw_cfg.get("iou_threshold", 0.45),
            device=yw_cfg.get("device", "cpu"),
            prompt_classes=yw_cfg.get("prompt_classes", ["cardboard box"]),
        )
    return BackgroundSubtractorDetector(
        method=det_cfg.get("background_subtractor", "MOG2"),
        min_contour_area=det_cfg.get("min_contour_area", 2000),
        max_contour_area=det_cfg.get("max_contour_area", 50000),
        blur_kernel=det_cfg.get("blur_kernel", 21),
        dilate_iterations=det_cfg.get("dilate_iterations", 2),
        history=det_cfg.get("history", 500),
        var_threshold=det_cfg.get("var_threshold", 50),
        detect_shadows=det_cfg.get("detect_shadows", False),
        learning_rate=det_cfg.get("learning_rate", 0.005),
        use_frame_diff=det_cfg.get("use_frame_diff", True),
        frame_diff_threshold=det_cfg.get("frame_diff_threshold", 25),
        motion_ratio_threshold=det_cfg.get("motion_ratio_threshold", 0.05),
        nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.3),
    )


# ── main class ────────────────────────────────────────────────────────────────

class ConveyorCounter:
    """
    Orchestrates the full detection → tracking → counting → logging pipeline.

    Responsibilities:
      - Initialise all subsystems from config.
      - Run the frame processing loop.
      - Handle keyboard input for runtime adjustments.
      - Gracefully shut down on quit or exception.
    """

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self.cfg = _load_config(config_path)
        _setup_logging(self.cfg)
        self.log = logging.getLogger(self.__class__.__name__)

        cam_cfg = self.cfg["camera"]
        perf_cfg = self.cfg.get("performance", {})
        trk_cfg = self.cfg["tracking"]
        cnt_cfg = self.cfg["counting"]
        disp_cfg = self.cfg["display"]
        log_cfg = self.cfg["logging"]

        self._frame_skip: int = perf_cfg.get("frame_skip", 0)
        self._resize_factor: float = perf_cfg.get("resize_factor", 1.0)
        self._show_mask: bool = disp_cfg.get("show_mask", False)

        source = cam_cfg.get("source", 0)
        # Disable the capture thread for video files — the thread races ahead
        # and skips frames, making tracking unreliable at high CPU speeds.
        is_video_file = isinstance(source, str)
        use_threading = False if is_video_file else perf_cfg.get("use_threading", True)

        self._camera = CameraCapture(
            source=source,
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            use_picamera=cam_cfg.get("use_picamera", False),
            use_threading=use_threading,
        )
        self._is_video_file = is_video_file

        self._detector = _build_detector(self.cfg)

        self._tracker = CentroidTracker(
            max_disappeared=trk_cfg.get("max_disappeared", 30),
            max_distance=trk_cfg.get("max_distance", 120),
        )

        proc_w = int(cam_cfg["width"] * self._resize_factor)
        proc_h = int(cam_cfg["height"] * self._resize_factor)
        self._counter = LineCounter(
            frame_width=proc_w,
            frame_height=proc_h,
            line_position=cnt_cfg.get("line_position", 0.5),
            orientation=cnt_cfg.get("line_orientation", "vertical"),
            direction=cnt_cfg.get("direction", "both"),
            min_speed=cnt_cfg.get("min_speed", 0.5),
            zone_width=cnt_cfg.get("zone_width", 0),
        )

        self._data_logger = DataLogger(
            csv_enabled=log_cfg.get("csv_enabled", True),
            sqlite_enabled=log_cfg.get("sqlite_enabled", True),
            csv_path=log_cfg.get("csv_path", "logs/counts.csv"),
            db_path=log_cfg.get("db_path", "logs/counts.db"),
        )

        self._display = Display(
            show_bounding_boxes=disp_cfg.get("show_bounding_boxes", True),
            show_tracking_ids=disp_cfg.get("show_tracking_ids", True),
            show_counting_line=disp_cfg.get("show_counting_line", True),
            show_fps=disp_cfg.get("show_fps", True),
            show_count=disp_cfg.get("show_count", True),
            line_color=disp_cfg.get("line_color", [0, 255, 0]),
            box_color=disp_cfg.get("box_color", [0, 165, 255]),
            counted_color=disp_cfg.get("counted_color", [0, 0, 255]),
            text_color=disp_cfg.get("text_color", [255, 255, 255]),
            save_snapshots=disp_cfg.get("save_snapshots", True),
            snapshot_dir=disp_cfg.get("snapshot_dir", "snapshots"),
        )

        self._fps_counter = FPSCounter(window_size=30)

        # Wire count events to the data logger
        self._counter.on_count(self._on_count_event)

        self._paused = False
        self._frame_index = 0
        self._frame_size_synced = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.log.info("Starting VisionCount pipeline")
        self._camera.start()

        # Determine inter-frame delay for playback throttling.
        # For video files use the file's native FPS; for live cameras use 1 ms
        # (the capture thread already paces frame delivery).
        native_fps = self._camera.actual_fps
        if self._is_video_file and native_fps > 0:
            self._frame_delay_ms = max(1, int(1000 / native_fps))
            self.log.info("Video FPS detected: %.1f — playback delay: %d ms",
                          native_fps, self._frame_delay_ms)
        else:
            self._frame_delay_ms = 1

        self._warm_up(frames=30)
        cv2.namedWindow("VisionCount", cv2.WINDOW_NORMAL)

        try:
            self._loop()
        except KeyboardInterrupt:
            self.log.info("Interrupted by user")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_frame_size(self, frame: np.ndarray) -> None:
        """Recompute counting geometry from the actual processed frame.

        Video files keep their native resolution regardless of the
        configured camera width/height, so the line/zone coordinates must
        come from the frames we actually process, not from the config.
        """
        h, w = frame.shape[:2]
        if (w, h) != (self._counter.frame_width, self._counter.frame_height):
            self.log.info(
                "Actual frame size %dx%d differs from configured %dx%d — "
                "recomputing counting line",
                w, h, self._counter.frame_width, self._counter.frame_height,
            )
            self._counter.update_frame_size(w, h)
        self._frame_size_synced = True

    def _warm_up(self, frames: int = 30) -> None:
        """Prepare the background model before counting starts.

        Video files: prime from a per-pixel median of frames sampled across
        the clip — no frames are consumed and objects near the start are not
        absorbed into the background.  Falls back to fast-learning warmup +
        rewind when the belt is too busy for a clean median.
        Live cameras: fast-learning warmup over the first frames (original
        behaviour).
        """
        if not hasattr(self._detector, "learning_rate"):
            return  # YOLO detectors need no background warmup

        if self._is_video_file and hasattr(self._detector, "prime"):
            from src.detection.background_subtractor import estimate_background
            samples = [self._preprocess(f) for f in self._camera.sample_frames(25)]
            if samples and not self._frame_size_synced:
                self._sync_frame_size(samples[0])
            background = estimate_background(samples) if samples else None
            if background is not None:
                self._detector.prime(background)
                self.log.info(
                    "Background primed from median of %d sampled frames",
                    len(samples),
                )
                return

        self.log.info("Warming up background model (%d frames)…", frames)
        delay = max(0.001, 1.0 / max(self._camera.actual_fps, 1))
        # Use a fast learning rate during warmup so the belt texture is
        # well-learned in just 30 frames.  0.05 → ~78 % saturation after 30
        # frames; the normal 0.001 gets only 3 % and leaves everything as
        # foreground at the start of the main loop.
        original_lr = self._detector.learning_rate
        self._detector.learning_rate = 0.05
        for i in range(frames):
            ok, frame = self._camera.read()
            if not ok or frame is None:
                if self._is_video_file:
                    self.log.warning(
                        "Video ended during warmup at frame %d/%d — "
                        "background model may be incomplete", i, frames
                    )
                break
            proc = self._preprocess(frame)
            if not self._frame_size_synced:
                self._sync_frame_size(proc)
            self._detector.detect(proc)
            time.sleep(delay)
        self._detector.learning_rate = original_lr
        # Replay the warmup frames for video files so crossings near the
        # start of the file are not silently consumed by warmup.
        if self._is_video_file and self._camera.rewind():
            self.log.info("Rewound video to frame 0 after warmup")

    def _loop(self) -> None:
        consecutive_empty = 0

        while True:
            # waitKey paces playback and polls keyboard.
            # For video files this is the primary throttle (frame_delay_ms ≈ 1/fps).
            key = cv2.waitKey(self._frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            self._handle_key(key, None)

            ok, frame = self._camera.read()
            if not ok or frame is None:
                consecutive_empty += 1
                if self._is_video_file:
                    # Video file ended
                    self.log.info(
                        "End of video. Final count: %d", self._counter.count
                    )
                    break
                if consecutive_empty > 30:
                    self.log.error("Camera not responding after 30 retries — exiting")
                    break
                time.sleep(0.01)
                continue
            consecutive_empty = 0

            if self._paused:
                cv2.imshow("VisionCount", frame)
                continue

            self._frame_index += 1
            if self._frame_skip > 0 and self._frame_index % (self._frame_skip + 1) != 0:
                continue

            proc_frame = self._preprocess(frame)
            if not self._frame_size_synced:
                self._sync_frame_size(proc_frame)
            detections, fg_mask = self._detector.detect(proc_frame)

            tracker_input = [(d.centroid, d.bbox) for d in detections]
            objects = self._tracker.update(tracker_input)

            # Every 60 frames print a one-line diagnostic so you can see
            # exactly what is being tracked and how far from the line each
            # object is — helps diagnose orientation/direction mismatches.
            if self._frame_index % 60 == 0 and objects:
                line = self._counter._line_coord
                orient = self._counter.orientation
                summaries = []
                for oid, obj in objects.items():
                    coord = obj.centroid[1] if orient == "horizontal" else obj.centroid[0]
                    summaries.append(
                        f"ID{oid}(coord={coord},line={line},spd={obj.speed:.1f})"
                    )
                self.log.info("Tracking %d object(s): %s", len(objects), "  ".join(summaries))

            newly_counted = self._counter.update(objects)

            annotated = self._display.render(
                proc_frame, objects, self._counter, self._fps_counter.fps, newly_counted
            )

            if self._show_mask and fg_mask is not None:
                mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                annotated = np.hstack([annotated, mask_bgr])

            cv2.imshow("VisionCount", annotated)
            self._fps_counter.tick()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self._resize_factor != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (int(w * self._resize_factor), int(h * self._resize_factor)),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def _handle_key(self, key: int, frame: Optional[np.ndarray]) -> None:
        if key == ord("r"):
            self._counter.reset()
            self._tracker.reset()
            self.log.info("Counter and tracker reset")
        elif key == ord("b"):
            if hasattr(self._detector, "reset_background"):
                self._detector.reset_background()
        elif key == ord("s") and frame is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("snapshots", f"manual_{ts}.jpg")
            os.makedirs("snapshots", exist_ok=True)
            cv2.imwrite(path, frame)
            self.log.info("Manual snapshot: %s", path)
        elif key == ord("p"):
            self._paused = not self._paused
            self.log.info("Paused: %s", self._paused)
        elif key == ord("+"):
            self._counter.line_position = min(
                1.0, self._counter.line_position + 0.02
            )
            self._counter._line_coord = self._counter._compute_line_coord()
            self.log.info("Line position: %.2f", self._counter.line_position)
        elif key == ord("-"):
            self._counter.line_position = max(
                0.0, self._counter.line_position - 0.02
            )
            self._counter._line_coord = self._counter._compute_line_coord()
            self.log.info("Line position: %.2f", self._counter.line_position)

    def _on_count_event(self, object_id: int, obj, total_count: int) -> None:
        self._data_logger.log(
            object_id=object_id,
            total_count=total_count,
            speed=obj.speed,
        )

    def _shutdown(self) -> None:
        self.log.info("Shutting down…")
        summary = self._data_logger.get_summary()
        if summary:
            self.log.info("Session summary: %s", summary)
        self._camera.stop()
        self._data_logger.close()
        cv2.destroyAllWindows()
        self.log.info("VisionCount stopped. Total count: %d", self._counter.count)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionCount — Conveyor Belt Object Counter"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--source",
        help="Override camera source (int index or video file path)",
    )
    parser.add_argument(
        "--method",
        choices=["background_subtraction", "yolo", "yolo_world"],
        help="Override detection method",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = args.config

    # Apply CLI overrides to the YAML before constructing the pipeline
    if args.source or args.method:
        cfg = _load_config(cfg_path)
        if args.source:
            try:
                cfg["camera"]["source"] = int(args.source)
            except ValueError:
                # Expand ~ and relative paths to absolute
                resolved = os.path.abspath(os.path.expanduser(args.source))
                if not os.path.isfile(resolved):
                    print(
                        f"[ERROR] Video file not found: {resolved}\n"
                        f"        Check the path and try again."
                    )
                    sys.exit(1)
                cfg["camera"]["source"] = resolved
        if args.method:
            cfg["detection"]["method"] = args.method
        # Write a temp override file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(cfg, tmp)
            cfg_path = tmp.name

    ConveyorCounter(config_path=cfg_path).run()


if __name__ == "__main__":
    main()
