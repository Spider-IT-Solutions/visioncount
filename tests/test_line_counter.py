"""Unit tests for LineCounter."""

import pytest
from collections import OrderedDict
from src.counting.line_counter import LineCounter
from src.tracking.centroid_tracker import TrackedObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj(obj_id: int, positions: list) -> TrackedObject:
    """Build a TrackedObject with a pre-populated centroid history."""
    obj = TrackedObject(obj_id, positions[0])
    for pos in positions[1:]:
        obj.update(pos)
    return obj


def _objects(*items) -> OrderedDict:
    od = OrderedDict()
    for obj in items:
        od[obj.object_id] = obj
    return od


# ---------------------------------------------------------------------------
# Horizontal line — top_to_bottom
# ---------------------------------------------------------------------------

class TestHorizontalTopToBottom:
    def _counter(self, **kw):
        return LineCounter(
            frame_width=640,
            frame_height=480,
            line_position=0.5,   # line at y=240
            orientation="horizontal",
            direction="top_to_bottom",
            min_speed=0.0,
            **kw,
        )

    def test_object_crossing_down_increments_count(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 230), (320, 250)])   # crosses y=240
        ctr.update(_objects(obj))
        assert ctr.count == 1

    def test_object_not_yet_crossed_does_not_count(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 220), (320, 235)])   # stays above line
        ctr.update(_objects(obj))
        assert ctr.count == 0

    def test_object_moving_up_does_not_count(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 250), (320, 230)])   # moves upward
        ctr.update(_objects(obj))
        assert ctr.count == 0

    def test_same_object_counted_only_once(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj))
        obj.update((320, 270))
        ctr.update(_objects(obj))
        assert ctr.count == 1

    def test_two_objects_each_counted_once(self):
        ctr = self._counter()
        obj0 = _make_obj(0, [(100, 230), (100, 260)])
        obj1 = _make_obj(1, [(400, 228), (400, 255)])
        ctr.update(_objects(obj0, obj1))
        assert ctr.count == 2


# ---------------------------------------------------------------------------
# Vertical line — left_to_right
# ---------------------------------------------------------------------------

class TestVerticalLeftToRight:
    def _counter(self):
        return LineCounter(
            frame_width=640,
            frame_height=480,
            line_position=0.5,   # line at x=320
            orientation="vertical",
            direction="left_to_right",
            min_speed=0.0,
        )

    def test_crossing_right_counts(self):
        ctr = self._counter()
        obj = _make_obj(0, [(310, 240), (330, 240)])
        ctr.update(_objects(obj))
        assert ctr.count == 1

    def test_crossing_left_does_not_count(self):
        ctr = self._counter()
        obj = _make_obj(0, [(330, 240), (310, 240)])
        ctr.update(_objects(obj))
        assert ctr.count == 0


# ---------------------------------------------------------------------------
# "both" directions
# ---------------------------------------------------------------------------

class TestBothDirections:
    def _counter(self):
        return LineCounter(
            frame_width=640,
            frame_height=480,
            line_position=0.5,
            orientation="horizontal",
            direction="both",
            min_speed=0.0,
        )

    def test_downward_counts(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 230), (320, 255)])
        ctr.update(_objects(obj))
        assert ctr.count == 1

    def test_upward_counts(self):
        ctr = self._counter()
        obj = _make_obj(0, [(320, 255), (320, 230)])
        ctr.update(_objects(obj))
        assert ctr.count == 1


# ---------------------------------------------------------------------------
# Min speed guard
# ---------------------------------------------------------------------------

class TestMinSpeed:
    def test_slow_object_not_counted(self):
        ctr = LineCounter(640, 480, 0.5, min_speed=10.0)
        # Move only 1 px — speed < 10
        obj = _make_obj(0, [(320, 239), (320, 241)])
        ctr.update(_objects(obj))
        assert ctr.count == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_zeroes_count(self):
        ctr = LineCounter(640, 480, 0.5, min_speed=0.0)
        obj = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj))
        ctr.reset()
        assert ctr.count == 0

    def test_reset_allows_recounting_same_id(self):
        ctr = LineCounter(640, 480, 0.5, orientation="horizontal",
                          direction="top_to_bottom", min_speed=0.0)
        obj = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj))
        ctr.reset()
        obj2 = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj2))
        assert ctr.count == 1


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TestCallback:
    def test_callback_called_on_count(self):
        events = []
        ctr = LineCounter(640, 480, 0.5, orientation="horizontal",
                          direction="top_to_bottom", min_speed=0.0)
        ctr.on_count(lambda obj_id, obj, total: events.append((obj_id, total)))
        obj = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj))
        assert events == [(0, 1)]

    def test_faulty_callback_does_not_crash_pipeline(self):
        ctr = LineCounter(640, 480, 0.5, orientation="horizontal",
                          direction="top_to_bottom", min_speed=0.0)
        ctr.on_count(lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
        obj = _make_obj(0, [(320, 230), (320, 260)])
        ctr.update(_objects(obj))   # must not raise
        assert ctr.count == 1
