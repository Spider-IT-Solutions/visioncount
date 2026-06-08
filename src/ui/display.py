"""
Display Renderer
================
Draws bounding boxes, tracking IDs, the counting line, FPS, and count
on a copy of the current frame.  Optionally saves a JPEG snapshot to
disk whenever a new object is counted.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Semi-transparent overlay alpha for the stats panel
_OVERLAY_ALPHA = 0.55


class Display:
    """
    Renders all visual overlays onto a BGR frame.

    All colour tuples are BGR (not RGB) to match OpenCV convention.
    """

    def __init__(
        self,
        show_bounding_boxes: bool = True,
        show_tracking_ids: bool = True,
        show_counting_line: bool = True,
        show_fps: bool = True,
        show_count: bool = True,
        line_color: Tuple = (0, 255, 0),
        box_color: Tuple = (0, 165, 255),
        counted_color: Tuple = (0, 0, 255),
        text_color: Tuple = (255, 255, 255),
        save_snapshots: bool = False,
        snapshot_dir: str = "snapshots",
    ) -> None:
        self.show_bounding_boxes = show_bounding_boxes
        self.show_tracking_ids = show_tracking_ids
        self.show_counting_line = show_counting_line
        self.show_fps = show_fps
        self.show_count = show_count
        self.line_color = tuple(line_color)
        self.box_color = tuple(box_color)
        self.counted_color = tuple(counted_color)
        self.text_color = tuple(text_color)
        self.save_snapshots = save_snapshots
        self.snapshot_dir = snapshot_dir

        if save_snapshots:
            os.makedirs(snapshot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        objects: dict,
        counter,
        fps: float,
        newly_counted: Optional[Dict[int, bool]] = None,
    ) -> np.ndarray:
        """
        Draw all overlays on a copy of *frame* and return the annotated image.

        Args:
            frame:         Raw BGR frame from the camera.
            objects:       OrderedDict[int, TrackedObject] from CentroidTracker.
            counter:       LineCounter instance (provides line coordinates and count).
            fps:           Current processing frame rate.
            newly_counted: Dict of {object_id: True} for objects just counted.
        """
        out = frame.copy()
        if newly_counted is None:
            newly_counted = {}

        if self.show_counting_line:
            self._draw_counting_line(out, counter)

        for obj_id, obj in objects.items():
            color = self.counted_color if obj.counted else self.box_color
            self._draw_object(out, obj_id, obj, color)

        self._draw_stats(out, counter.count, fps)

        if self.save_snapshots and newly_counted:
            obj_id = next(iter(newly_counted))
            self._save_snapshot(out, obj_id)

        return out

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_counting_line(self, frame: np.ndarray, counter) -> None:
        cv2.line(frame, counter.line_start, counter.line_end, self.line_color, 2)
        label_pos = (counter.line_start[0] + 6, counter.line_start[1] - 6)
        cv2.putText(
            frame, "COUNT LINE", label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.line_color, 1, cv2.LINE_AA,
        )

    def _draw_object(self, frame: np.ndarray, obj_id: int, obj, color) -> None:
        # Bounding box
        if self.show_bounding_boxes and obj.bbox is not None:
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Centroid dot
        cx, cy = obj.centroid
        cv2.circle(frame, (cx, cy), 4, color, -1)

        # Trajectory (last N positions)
        if len(obj.centroids) > 1:
            pts = np.array(obj.centroids, dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, 1, cv2.LINE_AA)

        # Label
        if self.show_tracking_ids:
            label = f"ID:{obj_id}"
            if obj.counted:
                label += " ✓"
            cv2.putText(
                frame, label, (cx - 10, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

    def _draw_stats(self, frame: np.ndarray, count: int, fps: float) -> None:
        h, w = frame.shape[:2]

        if self.show_count:
            text = f"Count: {count}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            # Semi-transparent backing rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - tw - 18, 5), (w - 5, th + 18), (0, 0, 0), -1)
            cv2.addWeighted(overlay, _OVERLAY_ALPHA, frame, 1 - _OVERLAY_ALPHA, 0, frame)
            cv2.putText(
                frame, text, (w - tw - 12, th + 11),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, self.text_color, 2, cv2.LINE_AA,
            )

        if self.show_fps:
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, frame: np.ndarray, object_id: int) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.snapshot_dir, f"count_{ts}_id{object_id}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Snapshot saved: %s", path)
