"""Unit tests for CentroidTracker."""

import pytest
from src.tracking.centroid_tracker import CentroidTracker, TrackedObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(x, y, w=20, h=20):
    """Make a (centroid, bbox) detection tuple."""
    return ((x + w // 2, y + h // 2), (x, y, w, h))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_single_detection_registers_id_zero(self):
        tracker = CentroidTracker()
        objects = tracker.update([_det(100, 100)])
        assert len(objects) == 1
        assert 0 in objects

    def test_two_detections_get_distinct_ids(self):
        tracker = CentroidTracker()
        objects = tracker.update([_det(10, 10), _det(300, 300)])
        assert len(objects) == 2
        assert set(objects.keys()) == {0, 1}

    def test_ids_are_monotonically_increasing(self):
        # Frame 1 registers object 0 at (10,10).
        # Frame 2 matches it to the same centroid and registers (210,210) as ID 1.
        tracker = CentroidTracker()
        tracker.update([_det(0, 0)])
        tracker.update([_det(0, 0), _det(200, 200)])
        ids = set(tracker.objects.keys())
        assert ids == {0, 1}
        assert tracker._next_id == 2


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class TestMatching:
    def test_close_detection_reuses_id(self):
        tracker = CentroidTracker(max_distance=50)
        tracker.update([_det(100, 100)])
        obj_id = list(tracker.objects.keys())[0]
        tracker.update([_det(105, 103)])   # moved 5 px — within max_distance
        assert obj_id in tracker.objects

    def test_far_detection_registers_new_id(self):
        tracker = CentroidTracker(max_distance=30)
        tracker.update([_det(100, 100)])
        tracker.update([_det(500, 500)])   # 566 px away — beyond max_distance
        assert len(tracker.objects) == 2

    def test_velocity_is_computed(self):
        tracker = CentroidTracker(max_distance=100)
        tracker.update([_det(100, 100)])
        tracker.update([_det(110, 100)])
        obj = list(tracker.objects.values())[0]
        assert obj.velocity[0] == pytest.approx(10, abs=1)


# ---------------------------------------------------------------------------
# Disappearance
# ---------------------------------------------------------------------------

class TestDisappearance:
    def test_object_disappears_after_max_frames(self):
        tracker = CentroidTracker(max_disappeared=3)
        tracker.update([_det(100, 100)])
        for _ in range(4):   # 4 > max_disappeared=3
            tracker.update([])
        assert len(tracker.objects) == 0

    def test_object_survives_within_max_frames(self):
        tracker = CentroidTracker(max_disappeared=5)
        tracker.update([_det(100, 100)])
        for _ in range(4):   # 4 <= max_disappeared=5
            tracker.update([])
        assert len(tracker.objects) == 1

    def test_reappearing_object_resets_disappeared_counter(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        tracker.update([_det(100, 100)])
        tracker.update([])
        tracker.update([_det(105, 100)])
        obj = list(tracker.objects.values())[0]
        assert obj.disappeared == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_objects(self):
        tracker = CentroidTracker()
        tracker.update([_det(10, 10), _det(200, 200)])
        tracker.reset()
        assert len(tracker.objects) == 0
        assert tracker._next_id == 0

    def test_ids_restart_after_reset(self):
        tracker = CentroidTracker()
        tracker.update([_det(0, 0)])
        tracker.reset()
        tracker.update([_det(0, 0)])
        assert list(tracker.objects.keys()) == [0]


# ---------------------------------------------------------------------------
# Speed / trajectory
# ---------------------------------------------------------------------------

class TestTrackedObject:
    def test_speed_zero_at_creation(self):
        obj = TrackedObject(0, (50, 50))
        assert obj.speed == 0.0

    def test_trajectory_capped_at_30(self):
        obj = TrackedObject(0, (0, 0))
        for i in range(50):
            obj.update((i, 0))
        assert len(obj.centroids) <= 30
