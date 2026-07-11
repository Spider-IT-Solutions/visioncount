import cv2
import numpy as np


def _odd(k):
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _gamma_lut(gamma_value):
    inv_gamma = 1.0 / max(0.01, gamma_value)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return table


def _apply_lab_l_channel(frame, fn):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = fn(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


_MORPH_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "erode": None,
    "dilate": None,
}


def apply_preprocessing(frame, cfg):
    """Fixed pipeline order: denoise -> blur variants -> CLAHE -> hist-eq ->
    sharpen -> gamma -> morphology. Each stage is independently gated by its
    own 'enabled' flag so effects can be combined or isolated for calibration."""
    out = frame

    denoise = cfg.get("denoise", {})
    if denoise.get("enabled"):
        strength = float(denoise.get("strength", 10))
        out = cv2.fastNlMeansDenoisingColored(out, None, strength, strength, 7, 21)

    blur = cfg.get("blur", {})
    if blur.get("enabled"):
        k = _odd(blur.get("kernel", 3))
        out = cv2.blur(out, (k, k))

    gaussian = cfg.get("gaussian_blur", {})
    if gaussian.get("enabled"):
        k = _odd(gaussian.get("kernel", 3))
        out = cv2.GaussianBlur(out, (k, k), gaussian.get("sigma", 0))

    median = cfg.get("median_blur", {})
    if median.get("enabled"):
        k = _odd(median.get("kernel", 3))
        out = cv2.medianBlur(out, k)

    clahe = cfg.get("clahe", {})
    if clahe.get("enabled"):
        clipper = cv2.createCLAHE(
            clipLimit=float(clahe.get("clip_limit", 2.0)),
            tileGridSize=(int(clahe.get("tile_grid", 8)), int(clahe.get("tile_grid", 8))),
        )
        out = _apply_lab_l_channel(out, clipper.apply)

    if cfg.get("hist_eq", {}).get("enabled"):
        out = _apply_lab_l_channel(out, cv2.equalizeHist)

    sharpen = cfg.get("sharpen", {})
    if sharpen.get("enabled") and sharpen.get("amount", 0) > 0:
        amount = sharpen["amount"] / 100.0
        blurred = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1 + amount, blurred, -amount, 0)

    gamma = cfg.get("gamma", {})
    if gamma.get("enabled") and gamma.get("value", 1.0) != 1.0:
        out = cv2.LUT(out, _gamma_lut(gamma.get("value", 1.0)))

    morph = cfg.get("morphology", {})
    if morph.get("enabled"):
        k = _odd(morph.get("kernel", 3))
        kernel = np.ones((k, k), np.uint8)
        iterations = max(1, int(morph.get("iterations", 1)))
        op = morph.get("operation", "open")
        if op == "erode":
            out = cv2.erode(out, kernel, iterations=iterations)
        elif op == "dilate":
            out = cv2.dilate(out, kernel, iterations=iterations)
        else:
            out = cv2.morphologyEx(out, _MORPH_OPS.get(op, cv2.MORPH_OPEN), kernel, iterations=iterations)

    return out
