import cv2
import numpy as np

from detection.base import Detection, Detector
from detection.nms import non_max_suppression

_CLEANUP_KERNEL = np.ones((3, 3), np.uint8)


class HsvContourDetector(Detector):
    """Multi-profile color-threshold + contour detection. Each enabled color
    profile gets its own HSV mask, cleaned independently (open + dilate to
    remove speckle noise), then all profile masks are OR-combined into one
    mask that contours are found on — one findContours call regardless of
    profile count, and touching regions from different profiles naturally
    merge into a single contour instead of producing separate boxes.

    No real confidence score exists for this method: for each surviving
    contour, whichever profile's own mask has the highest fill-ratio inside
    that bounding box wins — its name becomes the detection's label and its
    fill-ratio becomes the confidence (tight, solid-color blobs score near
    1.0; noisy/partial blobs score lower).

    A final IoU/NMS pass (detection/nms.py) collapses any remaining duplicate
    detections of the same physical object — e.g. a contour that fragmented
    into two pieces from a shadow, or a region genuinely matched by more than
    one profile's mask.
    """

    def detect(self, frame_bgr, detection_cfg, color_cfg, filters_cfg):
        profiles = [
            p for p in color_cfg.get("profiles", [])
            if p.get("enabled", True) and p.get("color_space", "hsv") == "hsv"
        ]

        h, w = frame_bgr.shape[:2]
        if not profiles:
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [], {"mask": empty_mask, "hsv_result": np.zeros_like(frame_bgr)}

        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        profile_masks = {}
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for p in profiles:
            lower = np.array(p["lower_hsv"])
            upper = np.array(p["upper_hsv"])
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _CLEANUP_KERNEL)
            mask = cv2.dilate(mask, _CLEANUP_KERNEL, iterations=1)
            profile_masks[p["id"]] = mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        hsv_result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=combined_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        min_area = filters_cfg["min_area"]
        max_area = filters_cfg["max_area"]
        min_w, max_w = filters_cfg["min_width"], filters_cfg["max_width"]
        min_h, max_h = filters_cfg["min_height"], filters_cfg["max_height"]
        aspect_min, aspect_max = filters_cfg["aspect_min"], filters_cfg["aspect_max"]
        conf_threshold = detection_cfg["confidence_threshold"]
        nms_threshold = detection_cfg.get("nms_threshold", 0.4)

        raw_detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
                continue
            aspect = bw / float(bh) if bh else 0
            if aspect < aspect_min or aspect > aspect_max:
                continue

            best_profile, best_fill = None, 0.0
            for p in profiles:
                sub = profile_masks[p["id"]][y : y + bh, x : x + bw]
                fill = float(np.count_nonzero(sub)) / float(bw * bh) if bw * bh else 0.0
                if fill > best_fill:
                    best_fill, best_profile = fill, p

            confidence = round(min(1.0, best_fill), 3)
            if confidence < conf_threshold:
                continue

            label = best_profile["name"] if best_profile else "object"
            raw_detections.append(Detection(x=x, y=y, w=bw, h=bh, confidence=confidence, label=label))

        detections = non_max_suppression(raw_detections, nms_threshold)
        return detections, {"mask": combined_mask, "hsv_result": hsv_result}
