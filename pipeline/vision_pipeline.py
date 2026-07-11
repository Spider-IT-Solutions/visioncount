import cv2

from counting.line_counter import LineCounter
from detection.base import Detection
from detection.registry import get_detector
from imaging.adjust import apply_camera_adjustments
from imaging.preprocessing import apply_preprocessing
from roi.roi_manager import RoiManager
from tracking.centroid_tracker import CentroidTracker


class VisionPipeline:
    """The full detect -> filter -> ROI gate -> track -> count -> annotate
    pipeline, extracted so the Studio's threaded wrapper (pipeline/processor.py)
    and a project's standalone runtime.py can share one implementation — a
    fix here benefits every deployed project without regenerating anything."""

    def __init__(self):
        self.tracker = CentroidTracker()
        self.line_counter = LineCounter()

    def reset_counts(self):
        self.line_counter.reset()

    def reset_all(self):
        """Full reset for a project switch — fresh tracker (drop stale IDs
        from the previous project) plus fresh counts."""
        self.tracker = CentroidTracker()
        self.line_counter.reset()

    def process(self, frame, cfg):
        """Returns (annotated_frame, tracks, events, stats_partial, debug)."""
        camera_cfg = cfg["camera"]
        adjusted = apply_camera_adjustments(frame, camera_cfg)
        adjusted = apply_preprocessing(adjusted, cfg["preprocessing"])
        adjusted = self._apply_rotation_flip(adjusted, camera_cfg)

        detector = get_detector(cfg["detection"]["backend"])
        resize_factor = cfg.get("performance", {}).get("resize_factor", 1.0) or 1.0
        if resize_factor != 1.0:
            small = cv2.resize(adjusted, None, fx=resize_factor, fy=resize_factor)
            detections, debug = detector.detect(small, cfg["detection"], cfg["color_calibration"], cfg["filters"])
            detections = [
                Detection(
                    x=int(d.x / resize_factor), y=int(d.y / resize_factor),
                    w=int(d.w / resize_factor), h=int(d.h / resize_factor),
                    confidence=d.confidence,
                )
                for d in detections
            ]
            full_size = (adjusted.shape[1], adjusted.shape[0])
            if "mask" in debug:
                debug["mask"] = cv2.resize(debug["mask"], full_size)
            if "hsv_result" in debug:
                debug["hsv_result"] = cv2.resize(debug["hsv_result"], full_size)
        else:
            detections, debug = detector.detect(adjusted, cfg["detection"], cfg["color_calibration"], cfg["filters"])

        roi_cfg = cfg["roi"]
        gated = [d for d in detections if RoiManager.contains(roi_cfg, d.x + d.w // 2, d.y + d.h // 2)]

        tracking_cfg = cfg["tracking"]
        self.tracker.max_distance = tracking_cfg.get("max_distance", 35)
        self.tracker.max_missed = tracking_cfg.get("object_timeout", 10)
        tracks = self.tracker.update(gated)

        lines_cfg = cfg["counting"]["lines"]
        events = self.line_counter.update(tracks, lines_cfg)

        annotated = self._annotate(adjusted.copy(), roi_cfg, lines_cfg, tracks)

        stats_partial = {
            "object_count_current": len(tracks),
            "counts": self.line_counter.get_counts(lines_cfg),
            "avg_confidence": round(sum(t.confidence for t in tracks) / len(tracks), 3) if tracks else 0.0,
        }
        return annotated, tracks, events, stats_partial, debug

    def _apply_rotation_flip(self, frame, camera_cfg):
        rotation = camera_cfg.get("rotation", 0) % 360
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        flip = camera_cfg.get("flip", "none")
        if flip == "horizontal":
            frame = cv2.flip(frame, 1)
        elif flip == "vertical":
            frame = cv2.flip(frame, 0)
        elif flip == "both":
            frame = cv2.flip(frame, -1)
        return frame

    def _annotate(self, frame, roi_cfg, lines_cfg, tracks):
        for shape in roi_cfg.get("shapes", []):
            if shape.get("enabled", True) and shape.get("type") == "rect":
                x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        counts = self.line_counter.get_counts(lines_cfg)
        for line in lines_cfg:
            if not line.get("enabled", True):
                continue
            p1, p2 = (line["x1"], line["y1"]), (line["x2"], line["y2"])
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            per_line = counts["per_line"].get(line["id"], {})
            label_text = " ".join(f"{label}:{n}" for label, n in per_line.items())
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(frame, label_text, (mid[0] + 6, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for t in tracks:
            cv2.rectangle(frame, (t.x, t.y), (t.x + t.w, t.y + t.h), (0, 255, 0), 2)
            cv2.circle(frame, t.center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {t.id} {t.confidence:.2f}", (t.x, max(0, t.y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'IN:{counts["total_in"]} OUT:{counts["total_out"]} TOTAL:{counts["total"]}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
