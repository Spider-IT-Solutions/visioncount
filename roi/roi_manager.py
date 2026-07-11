class RoiManager:
    """roi.shapes is a list of {type, enabled, x, y, w, h, ...}. A point passes
    the gate if there are no enabled rect shapes at all (no gating configured),
    or if it falls inside ANY enabled rect shape (multi-ROI via OR logic).
    type:"polygon" shapes are stored but not evaluated (Phase 3)."""

    @staticmethod
    def contains(roi_cfg, cx, cy):
        rects = [s for s in roi_cfg.get("shapes", []) if s.get("enabled", True) and s.get("type") == "rect"]
        if not rects:
            return True
        for shape in rects:
            x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
            if x <= cx <= x + w and y <= cy <= y + h:
                return True
        return False
