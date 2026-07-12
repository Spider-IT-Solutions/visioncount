def iou(a, b):
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold):
    """Greedy NMS: highest-confidence detection wins each overlapping cluster.
    Used to collapse the same physical object matched by more than one color
    profile (or fragmented into adjacent contours by shadows/reflections) down
    to a single detection. Reused as-is by any future real model backend —
    this is exactly what detection.nms_threshold was added for."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    for det in ordered:
        if all(iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept
