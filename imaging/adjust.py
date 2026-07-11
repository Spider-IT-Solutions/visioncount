import cv2
import numpy as np


def apply_camera_adjustments(frame, camera_cfg):
    """Software emulation of camera-level controls (brightness/contrast/
    saturation/exposure/gain) — applied regardless of source type since none
    of file/USB/RTSP sources here expose real hardware controls uniformly."""
    brightness = camera_cfg.get("brightness", 0)
    contrast = camera_cfg.get("contrast", 0)
    saturation = camera_cfg.get("saturation", 0)
    exposure = camera_cfg.get("exposure", 0)
    gain = camera_cfg.get("gain", 0)

    alpha = (1 + contrast / 100.0) * (1 + exposure / 100.0) * (1 + gain / 100.0)
    out = cv2.convertScaleAbs(frame, alpha=max(0.1, alpha), beta=brightness)

    if saturation != 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation / 100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out
