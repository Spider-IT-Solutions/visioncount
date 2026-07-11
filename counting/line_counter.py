def _side(x1, y1, x2, y2, px, py):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class LineCounter:
    """Multi-line crossing counter. A line is just two draggable points, so
    horizontal/vertical/angled all use the same signed-distance crossing test —
    no special-casing per orientation. Direction is derived from which side of
    the line a track's centroid was on the previous frame vs now.
    """

    def __init__(self):
        self._session = {}  # line_id -> {"forward": n, "backward": n}

    def reset(self):
        self._session = {}

    def _counters(self, line_id):
        return self._session.setdefault(line_id, {"forward": 0, "backward": 0})

    def update(self, tracks, lines_cfg):
        """Returns list of crossing events: [{"line_id", "track_id", "label"}]"""
        events = []
        for line in lines_cfg:
            if not line.get("enabled", True):
                continue
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            counters = self._counters(line["id"])

            for track in tracks:
                if track.prev_center is None:
                    continue
                px, py = track.prev_center
                cx, cy = track.center
                s_prev = _side(x1, y1, x2, y2, px, py)
                s_curr = _side(x1, y1, x2, y2, cx, cy)
                if s_prev == 0 or s_curr == 0:
                    continue
                if s_prev > 0 and s_curr < 0:
                    counters["forward"] += 1
                    events.append(
                        {"line_id": line["id"], "track_id": track.id, "label": line["forward_label"]}
                    )
                elif s_prev < 0 and s_curr > 0:
                    counters["backward"] += 1
                    events.append(
                        {"line_id": line["id"], "track_id": track.id, "label": line["backward_label"]}
                    )
        return events

    def get_counts(self, lines_cfg):
        per_line = {}
        total_in = 0
        total_out = 0
        for line in lines_cfg:
            counters = self._counters(line["id"])
            fwd, bwd = counters["forward"], counters["backward"]
            per_line[line["id"]] = {
                line["forward_label"]: fwd,
                line["backward_label"]: bwd,
            }
            if line["forward_label"] == "IN":
                total_in += fwd
            elif line["forward_label"] == "OUT":
                total_out += fwd
            if line["backward_label"] == "IN":
                total_in += bwd
            elif line["backward_label"] == "OUT":
                total_out += bwd
        return {
            "per_line": per_line,
            "total_in": total_in,
            "total_out": total_out,
            "total": total_in + total_out,
        }
