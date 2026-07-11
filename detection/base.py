from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str = "object"


class Detector(ABC):
    """Interface every detection backend must implement (hsv_contour today,
    mobilenet_ssd/yolov8/yolov11/custom_tflite are future Phase 3 implementations
    registered the same way in detection/registry.py)."""

    @abstractmethod
    def detect(self, frame_bgr, detection_cfg, color_cfg, filters_cfg):
        """Return (detections: list[Detection], debug: dict).
        detection_cfg = cfg["detection"], color_cfg = cfg["color_calibration"],
        filters_cfg = cfg["filters"]. debug may include intermediate frames
        (e.g. 'mask', 'hsv_result') used by the calibration panel; backends
        without such debug views return {}.
        """
        raise NotImplementedError
