import cv2
import numpy as np

from detection.base import Detection, Detector


class HsvContourDetector(Detector):
    """Color-threshold + contour detection. No real confidence score exists for
    this method, so confidence is approximated as the fraction of the bounding
    box actually covered by mask pixels (tight, solid-color blobs score near 1.0;
    noisy/partial blobs score lower) — meant to be a useful signal, not a fake number.
    """

    def detect(self, frame_bgr, detection_cfg, color_cfg, filters_cfg):
        hsv_cfg = color_cfg["hsv"]
        lower = np.array([hsv_cfg["l_h"], hsv_cfg["l_s"], hsv_cfg["l_v"]])
        upper = np.array([hsv_cfg["u_h"], hsv_cfg["u_s"], hsv_cfg["u_v"]])

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        hsv_result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        min_area = filters_cfg["min_area"]
        max_area = filters_cfg["max_area"]
        min_w, max_w = filters_cfg["min_width"], filters_cfg["max_width"]
        min_h, max_h = filters_cfg["min_height"], filters_cfg["max_height"]
        aspect_min, aspect_max = filters_cfg["aspect_min"], filters_cfg["aspect_max"]
        conf_threshold = detection_cfg["confidence_threshold"]

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue
            aspect = w / float(h) if h else 0
            if aspect < aspect_min or aspect > aspect_max:
                continue

            roi_mask = mask[y : y + h, x : x + w]
            fill_ratio = float(np.count_nonzero(roi_mask)) / float(w * h) if w * h else 0.0
            confidence = round(min(1.0, fill_ratio), 3)
            if confidence < conf_threshold:
                continue

            detections.append(Detection(x=x, y=y, w=w, h=h, confidence=confidence))

        return detections, {"mask": mask, "hsv_result": hsv_result}
