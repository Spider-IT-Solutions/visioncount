"""
Centroid Tracker
================
Classic single-camera multi-object tracker based on centroid distance matching.

Algorithm (each frame):
  1. Compute Euclidean distance between every existing tracked centroid and
     every new detection centroid  →  distance matrix D.
  2. Greedy assignment: sort existing objects by their minimum distance to any
     detection, then pair each with its closest unmatched detection if the
     distance is within max_distance.
  3. Unmatched existing objects → increment disappeared counter.
     Exceeded max_disappeared → deregister.
  4. Unmatched new detections  → register as new objects.

Complexity is O(N·M) where N = tracked objects, M = new detections.
For conveyor belts (typically < 20 objects on screen) this is fast enough.
For dense scenes prefer DeepSORT or ByteTrack.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


class TrackedObject:
    """State for a single tracked object."""

    def __init__(self, object_id: int, centroid: Tuple[int, int]) -> None:
        self.object_id = object_id
        self.centroid: Tuple[int, int] = centroid
        # Rolling history for trajectory drawing (last 30 positions)
        self.centroids: List[Tuple[int, int]] = [centroid]
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.disappeared: int = 0
        self.counted: bool = False

    # ------------------------------------------------------------------

    def update(
        self,
        centroid: Tuple[int, int],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if self.centroids:
            prev = self.centroids[-1]
            # The object may have been invisible for `disappeared` frames, so
            # the measured delta spans (disappeared + 1) frames.  Divide to get
            # a true per-frame velocity, otherwise speed spikes after a gap.
            gap = self.disappeared + 1
            self.velocity = (
                float(centroid[0] - prev[0]) / gap,
                float(centroid[1] - prev[1]) / gap,
            )
        self.centroid = centroid
        self.centroids.append(centroid)
        if len(self.centroids) > 30:
            self.centroids.pop(0)
        if bbox is not None:
            self.bbox = bbox
        self.disappeared = 0

    def predict(self, frames_ahead: int) -> Tuple[float, float]:
        """Constant-velocity position estimate `frames_ahead` frames from the
        last measurement.  Conveyor motion is near-constant velocity, so this
        is a cheap stand-in for a Kalman prediction step."""
        return (
            self.centroid[0] + self.velocity[0] * frames_ahead,
            self.centroid[1] + self.velocity[1] * frames_ahead,
        )

    @property
    def speed(self) -> float:
        return float(np.hypot(self.velocity[0], self.velocity[1]))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrackedObject(id={self.object_id}, "
            f"centroid={self.centroid}, disappeared={self.disappeared})"
        )


# ---------------------------------------------------------------------------


class CentroidTracker:
    """
    Assigns stable integer IDs to moving objects across video frames.

    Args:
        max_disappeared: Frames a tracked object may be invisible before removal.
        max_distance:    Maximum centroid movement (px) between frames for a
                         match to be accepted.  Set lower for slow belts, higher
                         for fast ones or low frame rates.
    """

    def __init__(
        self,
        max_disappeared: int = 40,
        max_distance: int = 80,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id: int = 0
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[Tuple[int, int], Optional[Tuple[int, int, int, int]]]],
    ) -> "OrderedDict[int, TrackedObject]":
        """
        Update tracker state with new detections.

        Args:
            detections: list of (centroid, bbox) tuples produced by a detector.

        Returns:
            OrderedDict mapping object_id → TrackedObject (all live objects).
        """
        if not detections:
            for obj in list(self.objects.values()):
                obj.disappeared += 1
                if obj.disappeared > self.max_disappeared:
                    self._deregister(obj.object_id)
            return self.objects

        input_centroids = np.array([d[0] for d in detections], dtype="float")
        input_bboxes = [d[1] for d in detections]

        if not self.objects:
            for i, c in enumerate(input_centroids):
                self._register((int(c[0]), int(c[1])), input_bboxes[i])
            return self.objects

        object_ids = list(self.objects.keys())
        # Match against PREDICTED positions, not last-seen positions.  A box
        # that goes undetected for k frames keeps moving down the belt; its
        # frozen last position drifts max_distance away and the re-detection
        # would spawn a new ID (ID churn → double counts / missed counts).
        # Constant-velocity prediction keeps the gate centred on where the
        # box actually is.  For objects seen last frame this predicts one
        # frame ahead, which is equally correct.
        object_centroids = np.array(
            [o.predict(o.disappeared + 1) for o in self.objects.values()],
            dtype="float",
        )

        # Distance matrix: rows = existing objects, cols = new detections
        D = dist.cdist(object_centroids, input_centroids)

        # Sort rows by minimum distance so closest pairs are matched first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id = object_ids[row]
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.objects[obj_id].update(centroid, input_bboxes[col])
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        # Always increment disappeared for unmatched tracked objects…
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].disappeared += 1
            if self.objects[obj_id].disappeared > self.max_disappeared:
                self._deregister(obj_id)

        # …and always register unmatched new detections, regardless of which
        # side of the matrix is larger.  The classic pyimagesearch version
        # skips this when tracked >= new, silently dropping new arrivals.
        for col in unused_cols:
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self._register(centroid, input_bboxes[col])

        return self.objects

    def reset(self) -> None:
        """Clear all tracked objects and reset the ID counter."""
        self.objects.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(
        self,
        centroid: Tuple[int, int],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> int:
        obj = TrackedObject(self._next_id, centroid)
        obj.bbox = bbox
        self.objects[self._next_id] = obj
        object_id = self._next_id
        self._next_id += 1
        logger.debug("Registered object ID=%d at %s", object_id, centroid)
        return object_id

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        logger.debug("Deregistered object ID=%d", object_id)
