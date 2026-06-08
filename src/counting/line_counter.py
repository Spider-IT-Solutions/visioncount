"""
Virtual Line Counter
====================
Counts objects crossing a virtual line drawn across the frame.

Zone-based counting
-------------------
Instead of requiring the centroid to straddle the exact line pixel, this
counter uses a ZONE:

    |<----zone_width--->|<line>|<---zone_width---->|
    LEFT_BOUNDARY             RIGHT_BOUNDARY
       = line - zone_width       = line + zone_width

An object moving RIGHT→LEFT is counted the moment its centroid crosses the
RIGHT_BOUNDARY (enters the zone).  An object moving LEFT→RIGHT is counted
when it crosses the LEFT_BOUNDARY.

Why zone instead of exact pixel:
  • At 24 fps, a box moving at 40 px/frame can jump from x=185 to x=145 in
    one frame — skipping over a line at x=160 without a detection on each side.
  • Zone width of 50 means the "effective counting line" is 50 px before the
    visual line, making counting robust to fast boxes and sparse detections.

Ghost protection
----------------
Only objects that were actually updated THIS frame (obj.disappeared == 0)
are evaluated. Ghost objects (frozen at their last detected position) are
skipped, preventing the repeated "close but not crossing" spam and wrong
counts based on stale centroids.

Orientation:
  Belt moves LEFT ↔ RIGHT  →  orientation: "vertical"   (line runs top↕bottom)
  Belt moves UP   ↕ DOWN   →  orientation: "horizontal" (line runs left↔right)
"""

import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


class LineCounter:
    """Counts objects crossing a virtual counting zone."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        line_position: float = 0.5,
        orientation: str = "vertical",
        direction: str = "both",
        min_speed: float = 0.5,
        zone_width: int = 0,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.line_position = line_position
        self.orientation = orientation
        self.direction = direction
        self.min_speed = min_speed
        self.zone_width = zone_width

        self.count: int = 0
        self._counted_ids: set = set()
        self._on_count_callbacks: List[Callable] = []
        self._line_coord: int = self._compute_line_coord()

        logger.info(
            "LineCounter ready — orientation=%s  direction=%s  "
            "line_coord=%d  zone=[%d, %d]  min_speed=%.1f",
            orientation, direction,
            self._line_coord,
            self._line_coord - zone_width,
            self._line_coord + zone_width,
            min_speed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def line_start(self) -> Tuple[int, int]:
        if self.orientation == "horizontal":
            return (0, self._line_coord)
        return (self._line_coord, 0)

    @property
    def line_end(self) -> Tuple[int, int]:
        if self.orientation == "horizontal":
            return (self.frame_width, self._line_coord)
        return (self._line_coord, self.frame_height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_count(self, callback: Callable) -> None:
        """Register callable(object_id, tracked_object, total_count) fired on each count."""
        self._on_count_callbacks.append(callback)

    def update(self, objects: "OrderedDict") -> Dict[int, bool]:
        """
        Evaluate live (non-ghost) tracked objects for zone crossing.

        Returns dict mapping object_id → True for every object counted this frame.
        """
        newly_counted: Dict[int, bool] = {}

        for obj_id, obj in objects.items():
            if obj_id in self._counted_ids:
                continue

            # ── Skip ghost objects (centroid frozen at last detected position) ──
            if obj.disappeared > 0:
                continue

            # ── Need at least two positions ───────────────────────────────────
            if len(obj.centroids) < 2:
                logger.debug("ID=%d: waiting for 2nd centroid (%d so far)",
                             obj_id, len(obj.centroids))
                continue

            # ── Minimum speed guard ───────────────────────────────────────────
            if obj.speed < self.min_speed:
                logger.debug("ID=%d: speed=%.2f below min_speed=%.2f",
                             obj_id, obj.speed, self.min_speed)
                continue

            prev_cx, prev_cy = obj.centroids[-2]
            curr_cx, curr_cy = obj.centroid

            if self._check_zone_crossing(prev_cx, prev_cy, curr_cx, curr_cy):
                self.count += 1
                self._counted_ids.add(obj_id)
                obj.counted = True
                newly_counted[obj_id] = True
                logger.info(
                    "COUNTED  ID=%d  total=%d  coord=%d→%d  speed=%.1f px/frame",
                    obj_id, self.count,
                    prev_cy if self.orientation == "horizontal" else prev_cx,
                    curr_cy if self.orientation == "horizontal" else curr_cx,
                    obj.speed,
                )
                self._fire_callbacks(obj_id, obj)
            else:
                # Lightweight proximity debug (only for truly live objects)
                coord = curr_cy if self.orientation == "horizontal" else curr_cx
                prev_coord = prev_cy if self.orientation == "horizontal" else prev_cx
                if abs(coord - self._line_coord) < self.zone_width + 30:
                    logger.debug(
                        "ID=%d approaching — coord=%d  line=%d  zone=[%d,%d]  spd=%.1f",
                        obj_id, coord, self._line_coord,
                        self._line_coord - self.zone_width,
                        self._line_coord + self.zone_width,
                        obj.speed,
                    )

        return newly_counted

    def reset(self) -> None:
        self.count = 0
        self._counted_ids.clear()
        logger.info("Counter reset to 0")

    def update_frame_size(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height
        self._line_coord = self._compute_line_coord()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_line_coord(self) -> int:
        if self.orientation == "horizontal":
            return int(self.frame_height * self.line_position)
        return int(self.frame_width * self.line_position)

    def _check_zone_crossing(
        self,
        prev_x: int, prev_y: int,
        curr_x: int, curr_y: int,
    ) -> bool:
        """
        Count when a centroid crosses into the zone around the line.

        RIGHT_BOUNDARY = line_coord + zone_width  (entry point for right→left)
        LEFT_BOUNDARY  = line_coord - zone_width  (entry point for left→right)

        This is more robust than a strict line cross:
          • Handles fast boxes that skip over the exact line pixel in one frame
          • Handles boxes detected intermittently near the line
        """
        line = self._line_coord
        right_boundary = line + self.zone_width
        left_boundary = line - self.zone_width

        if self.orientation == "horizontal":
            prev_coord, curr_coord = prev_y, curr_y
        else:
            prev_coord, curr_coord = prev_x, curr_x

        if self.direction in ("top_to_bottom", "left_to_right"):
            # Crosses LEFT_BOUNDARY from outside (left→right)
            return prev_coord < left_boundary <= curr_coord

        if self.direction in ("bottom_to_top", "right_to_left"):
            # Crosses RIGHT_BOUNDARY from outside (right→left)
            return prev_coord > right_boundary >= curr_coord

        # "both" — either direction
        right_to_left = prev_coord > right_boundary >= curr_coord
        left_to_right = prev_coord < left_boundary <= curr_coord
        return right_to_left or left_to_right

    def _fire_callbacks(self, obj_id: int, obj) -> None:
        for cb in self._on_count_callbacks:
            try:
                cb(obj_id, obj, self.count)
            except Exception as exc:
                logger.error("Count callback raised an exception: %s", exc)
