"""
Interactive Calibration Tool
=============================
Helps you tune detection and counting parameters without editing YAML by hand.

Usage:
    python calibration/calibrate.py [--source 0] [--config config/settings.yaml]

Controls:
  Mouse click on the live frame   → set counting line position
  d                               → toggle detection mask overlay
  +/-                             → adjust contour area threshold (min)
  [/]                             → adjust contour area threshold (max)
  t                               → tune var_threshold (+/- 5)
  b                               → rebuild background model now
  s                               → save current parameters to config file
  q / ESC                         → quit without saving

The right panel shows:
  • Raw camera feed with detected contours
  • Foreground mask
  • Tuned parameter values

When you're happy, press 's' to write the tuned values back to settings.yaml.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from src.camera.capture import CameraCapture
from src.detection.background_subtractor import BackgroundSubtractorDetector


# ── State shared between callbacks and the main loop ──────────────────────────

class CalibState:
    def __init__(self, cfg: dict) -> None:
        det = cfg["detection"]
        cam = cfg["camera"]
        self.min_area: int = det.get("min_contour_area", 2000)
        self.max_area: int = det.get("max_contour_area", 50000)
        self.var_threshold: float = det.get("var_threshold", 50.0)
        self.blur_kernel: int = det.get("blur_kernel", 21)
        self.dilate_iter: int = det.get("dilate_iterations", 2)
        self.line_pos: float = cfg["counting"].get("line_position", 0.5)
        self.frame_h: int = cam.get("height", 480)
        self.frame_w: int = cam.get("width", 640)
        self.show_mask: bool = False
        self.rebuild_bg: bool = False


def _mouse_cb(event, x, y, flags, state: CalibState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.line_pos = y / state.frame_h
        print(f"  [CALIB] Line position set to {state.line_pos:.3f}")


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, detections, state: CalibState) -> np.ndarray:
    out = frame.copy()
    line_y = int(state.frame_h * state.line_pos)
    cv2.line(out, (0, line_y), (state.frame_w, line_y), (0, 255, 0), 2)
    cv2.putText(out, "COUNT LINE", (6, line_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(out, f"{int(det.area)}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    return out


def _draw_params(frame: np.ndarray, state: CalibState) -> np.ndarray:
    panel = np.zeros((frame.shape[0], 260, 3), dtype=np.uint8)
    params = [
        ("min_area", state.min_area, "[+] +100  [-] -100"),
        ("max_area", state.max_area, "[ ] +5000  [] -5000"),
        ("var_thresh", state.var_threshold, "[t] +5  [T] -5"),
        ("blur_kernel", state.blur_kernel, ""),
        ("dilate_iter", state.dilate_iter, ""),
        ("line_pos", f"{state.line_pos:.3f}", "click to set"),
    ]
    y = 30
    cv2.putText(panel, "PARAMETERS", (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for name, val, hint in params:
        cv2.putText(panel, f"{name}: {val}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if hint:
            cv2.putText(panel, hint, (10, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)
        y += 38
    cv2.putText(panel, "s = save  q = quit", (10, panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return np.hstack([frame, panel])


# ── Main calibration loop ─────────────────────────────────────────────────────

def run_calibration(cfg: dict, cfg_path: str) -> None:
    state = CalibState(cfg)
    cam_cfg = cfg["camera"]

    camera = CameraCapture(
        source=cam_cfg.get("source", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        use_threading=True,
    ).start()

    detector = BackgroundSubtractorDetector(
        min_contour_area=state.min_area,
        max_contour_area=state.max_area,
        var_threshold=state.var_threshold,
        blur_kernel=state.blur_kernel,
        dilate_iterations=state.dilate_iter,
    )

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", _mouse_cb, state)

    print("\n[CALIB] Click on frame to set line | s=save | q=quit\n")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            continue

        state.frame_h, state.frame_w = frame.shape[:2]

        if state.rebuild_bg:
            detector.reset_background()
            state.rebuild_bg = False

        # Rebuild detector if params changed
        detector.min_contour_area = state.min_area
        detector.max_contour_area = state.max_area

        detections, fg_mask = detector.detect(frame)

        annotated = _draw_overlay(frame, detections, state)

        if state.show_mask and fg_mask is not None:
            mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            annotated = np.hstack([annotated, mask_bgr])

        display = _draw_params(annotated, state)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("+"):
            state.min_area = min(state.min_area + 100, state.max_area - 100)
        elif key == ord("-"):
            state.min_area = max(100, state.min_area - 100)
        elif key == ord("]"):
            state.max_area += 5000
        elif key == ord("["):
            state.max_area = max(state.min_area + 100, state.max_area - 5000)
        elif key == ord("t"):
            state.var_threshold = min(200.0, state.var_threshold + 5)
        elif key == ord("T"):
            state.var_threshold = max(5.0, state.var_threshold - 5)
        elif key == ord("d"):
            state.show_mask = not state.show_mask
        elif key == ord("b"):
            state.rebuild_bg = True
        elif key == ord("s"):
            _save_params(cfg, cfg_path, state)
            print(f"[CALIB] Parameters saved to {cfg_path}")

    camera.stop()
    cv2.destroyAllWindows()


def _save_params(cfg: dict, cfg_path: str, state: CalibState) -> None:
    cfg["detection"]["min_contour_area"] = state.min_area
    cfg["detection"]["max_contour_area"] = state.max_area
    cfg["detection"]["var_threshold"] = float(state.var_threshold)
    cfg["counting"]["line_position"] = round(state.line_pos, 4)
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VisionCount Calibration Tool")
    parser.add_argument("--source", help="Camera source (int or file path)")
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Path to settings YAML"
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.source is not None:
        try:
            cfg["camera"]["source"] = int(args.source)
        except ValueError:
            cfg["camera"]["source"] = args.source

    run_calibration(cfg, args.config)


if __name__ == "__main__":
    main()
