import math
from dataclasses import dataclass


@dataclass
class Track:
    id: int
    x: int
    y: int
    w: int
    h: int
    confidence: float
    center: tuple
    prev_center: tuple  # None on the frame a track is first created


class CentroidTracker:
    """Nearest-centroid tracker (same approach as the original tracker.py) with
    added short occlusion tolerance (max_missed) so a track surviving a couple of
    missed frames near the counting line doesn't get a new ID and double-count.
    Same update(detections) -> list[Track] interface any future SORT/ByteTrack/
    DeepSORT backend should implement.
    """

    def __init__(self, max_distance=35, max_missed=10):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = {}  # id -> {"center", "prev_center", "missed"}
        self.next_id = 0

    def update(self, detections):
        assigned_ids = set()
        results = []

        for det in detections:
            cx = det.x + det.w // 2
            cy = det.y + det.h // 2

            best_id, best_dist = None, self.max_distance
            for tid, t in self.tracks.items():
                if tid in assigned_ids:
                    continue
                dist = math.hypot(cx - t["center"][0], cy - t["center"][1])
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is not None:
                prev_center = self.tracks[best_id]["center"]
                tid = best_id
            else:
                tid = self.next_id
                self.next_id += 1
                prev_center = None

            self.tracks[tid] = {"center": (cx, cy), "prev_center": prev_center, "missed": 0}
            assigned_ids.add(tid)

            results.append(
                Track(
                    id=tid,
                    x=det.x,
                    y=det.y,
                    w=det.w,
                    h=det.h,
                    confidence=det.confidence,
                    center=(cx, cy),
                    prev_center=prev_center,
                )
            )

        for tid in list(self.tracks.keys()):
            if tid not in assigned_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return results
