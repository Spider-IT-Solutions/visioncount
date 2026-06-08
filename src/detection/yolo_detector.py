"""
Optional YOLO-based detector (YOLOv8 / YOLOv11).

Install prerequisite:
    pip install ultralytics

On Raspberry Pi 4 the nano model (yolov8n.pt) runs ~3–5 FPS without hardware
acceleration.  For real-time performance use TFLite export + a Coral Edge TPU,
or quantise to INT8 via:
    yolo export model=yolov8n.pt format=tflite int8=True
"""

import numpy as np
import logging
from typing import List, Optional, Tuple

from .background_subtractor import Detection

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Drop-in replacement for BackgroundSubtractorDetector using Ultralytics YOLO.

    The detect() method returns the same (List[Detection], None) signature so
    the rest of the pipeline needs no changes.

    Args:
        model_path:   Path to a .pt or .tflite model file.
                      On first run with a .pt path ultralytics auto-downloads.
        confidence:   Minimum detection confidence (0–1).
        iou_threshold: NMS IoU threshold.
        classes:      COCO class IDs to keep, or None for all.
                      For generic boxes you may want None (all) and then filter
                      by size in the main loop.
        device:       "cpu", "0" (CUDA GPU 0), or "mps" (Apple Silicon).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.50,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        self._model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(model_path)
            self._model.to(self.device)
            logger.info("YOLO model loaded: %s on %s", model_path, self.device)
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO detection.\n"
                "Install with:  pip install ultralytics"
            ) from exc

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], None]:
        """Run inference on one BGR frame.  Returns (detections, None)."""
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                detections.append(Detection(
                    bbox=(x1, y1, w, h),
                    centroid=(cx, cy),
                    area=float(w * h),
                    confidence=float(box.conf[0]),
                ))

        return detections, None
