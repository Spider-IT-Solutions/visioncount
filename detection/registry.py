from detection.hsv_contour import HsvContourDetector

# None entries are Phase 3 backends: the dashboard's model-select dropdown lists
# them (disabled, tagged "Phase 3") to expose the seam without pretending they work.
DETECTOR_REGISTRY = {
    "hsv_contour": HsvContourDetector,
    "mobilenet_ssd": None,
    "yolov8": None,
    "yolov11": None,
    "custom_tflite": None,
}

_instances = {}


def get_detector(name):
    cls = DETECTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"detector backend '{name}' is not available yet")
    if name not in _instances:
        _instances[name] = cls()
    return _instances[name]


def available_backends():
    return [{"name": name, "available": cls is not None} for name, cls in DETECTOR_REGISTRY.items()]
