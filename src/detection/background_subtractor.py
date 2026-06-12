import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_background(
    frames: List[np.ndarray], stability_threshold: float = 6.0
) -> Optional[np.ndarray]:
    """Estimate a clean background as the per-pixel median of sampled frames.

    Moving objects vanish in the median as long as each pixel is object-free
    in >50% of the samples.  On busy belts (occupancy above 50%) the median
    smears object pixels into the background and priming with it would hide
    real objects — detected by comparing medians of the odd and even sample
    halves: a true background is stable across both, a contaminated one is
    not.  Returns None when contaminated; callers should fall back to
    frame-by-frame warmup.
    """
    if len(frames) < 4:
        return None
    stack = np.stack(frames)
    m1 = np.median(stack[::2], axis=0)
    m2 = np.median(stack[1::2], axis=0)
    diff = np.abs(m1.astype(int) - m2.astype(int)).mean()
    if diff > stability_threshold:
        logger.info(
            "Median background unstable (diff=%.1f > %.1f) — belt too busy, "
            "skipping median priming", diff, stability_threshold,
        )
        return None
    return np.median(stack, axis=0).astype("uint8")


@dataclass
class Detection:
    """One detected object — bounding box, centroid, area, and optional confidence."""
    bbox: Tuple[int, int, int, int]    # x, y, w, h  (top-left origin)
    centroid: Tuple[int, int]
    area: float
    confidence: float = 1.0


class BackgroundSubtractorDetector:
    """
    Detects MOVING objects only — static objects are ignored.

    Two-layer motion filter
    -----------------------
    Layer 1 — MOG2/KNN background subtraction
        Marks pixels that differ from the learned background model.
        Catches objects that just entered the frame or are actively moving.

    Layer 2 — Frame differencing
        Computes the absolute pixel difference between the current frame
        and the previous one.  Only pixels that are *changing right now*
        survive.  A stopped box on the belt produces zero diff → filtered out.

    Final mask = Layer1 AND Layer2
        A contour must be both "different from background" AND "currently
        moving" to generate a Detection.  This eliminates:
          • Belt texture false positives
          • Objects that stopped moving on the belt
          • Lighting-change ghosts that briefly appear in the BGS mask

    Per-contour motion check
    ------------------------
    Even after the AND-mask, a contour passes only if the fraction of its
    bounding-box pixels that are active in the frame-diff mask exceeds
    `motion_ratio_threshold`.  This removes large static blobs that have
    just a few noisy moving pixels inside them.

    Parameters
    ----------
    learning_rate : float
        Controls how fast MOG2 absorbs objects into the background.
        -1  → automatic (≈1/history).
        0.005 → a stopped box is absorbed after ~200 frames (~7 s at 30 fps).
        Lower values keep stopped boxes in the foreground longer.
    use_frame_diff : bool
        Set False to disable the frame-differencing layer (Layer 2).
        Useful when the camera itself vibrates and produces spurious diffs.
    frame_diff_threshold : int
        Pixel-level intensity change (0–255) that counts as motion in the
        frame-diff mask.  Increase for noisy/vibrating cameras.
    motion_ratio_threshold : float
        Fraction (0–1) of a contour's bounding-box pixels that must show
        frame-diff activity for the contour to be classified as moving.
        Default 0.05 means at least 5 % of the box must be actively moving.
    """

    def __init__(
        self,
        method: str = "MOG2",
        min_contour_area: int = 2000,
        max_contour_area: int = 50000,
        blur_kernel: int = 21,
        dilate_iterations: int = 2,
        history: int = 500,
        var_threshold: float = 50.0,
        detect_shadows: bool = False,
        learning_rate: float = 0.005,
        use_frame_diff: bool = True,
        frame_diff_threshold: int = 25,
        motion_ratio_threshold: float = 0.05,
        nms_iou_threshold: float = 0.3,
    ) -> None:
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.blur_kernel = blur_kernel
        self.dilate_iterations = dilate_iterations
        self.learning_rate = learning_rate
        self.use_frame_diff = use_frame_diff
        self.frame_diff_threshold = frame_diff_threshold
        self.motion_ratio_threshold = motion_ratio_threshold
        self.nms_iou_threshold = nms_iou_threshold

        self._method = method
        self._history = history
        self._var_threshold = var_threshold
        self._detect_shadows = detect_shadows

        self._bg_sub = self._build_subtractor()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self._prev_gray: Optional[np.ndarray] = None   # previous blurred frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        Run detection on one BGR frame.

        Returns
        -------
        detections : list[Detection]
        fg_mask    : binary mask showing confirmed moving regions (for debug)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # ── Layer 1: background subtraction ───────────────────────────
        bgs_mask = self._bg_sub.apply(blurred, learningRate=self.learning_rate)
        # Keep only definite foreground (255); discard shadows (127)
        _, bgs_mask = cv2.threshold(bgs_mask, 200, 255, cv2.THRESH_BINARY)

        # ── Layer 2: frame differencing (per-contour classifier) ───────
        # NOTE: deliberately NOT ANDed with the BGS mask.  A uniformly
        # coloured box only produces frame-diff at its edges (interior
        # pixels slide over identically coloured pixels), so an AND mask
        # reduces slow boxes to thin slivers that the erosion below wipes
        # out entirely.  Instead, contours come from the BGS mask alone and
        # the diff mask is used afterwards to classify each contour as
        # moving or static.
        diff_mask: Optional[np.ndarray] = None
        if self.use_frame_diff and self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, blurred)
            _, diff_mask = cv2.threshold(
                diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY
            )

        self._prev_gray = blurred  # save for next frame

        # ── Morphological cleanup ──────────────────────────────────────
        combined = cv2.erode(bgs_mask, self._kernel, iterations=1)
        combined = cv2.dilate(combined, self._kernel, iterations=self.dilate_iterations)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Per-contour motion ratio check ────────────────────────────
        # Checked against the RAW diff mask: a stopped box has ~zero diff
        # anywhere in its bbox and is rejected; a slow box still shows
        # diff along its edges, which is enough to pass a small threshold.
        ratio_mask = diff_mask

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_contour_area <= area <= self.max_contour_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Verify the contour contains enough actively-moving pixels
            if ratio_mask is not None:
                roi = ratio_mask[y: y + h, x: x + w]
                if roi.size > 0:
                    motion_ratio = np.count_nonzero(roi) / roi.size
                    if motion_ratio < self.motion_ratio_threshold:
                        continue   # mostly static — skip

            cx, cy = x + w // 2, y + h // 2
            detections.append(Detection(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
            ))

        if self.nms_iou_threshold > 0 and len(detections) > 1:
            detections = self._nms(detections, self.nms_iou_threshold)

        return detections, combined

    def reset_background(self) -> None:
        """Rebuild background model and clear frame-diff state."""
        self._bg_sub = self._build_subtractor()
        self._prev_gray = None
        logger.info("Background model reset")

    def prime(self, background: np.ndarray) -> None:
        """Initialise the background model from a clean background image.

        Intended for video files: build a per-pixel median over frames
        sampled across the whole clip (objects move, the belt does not, so
        the median is an object-free belt) and prime the model with it.
        Avoids the warmup dilemma where either the first N frames are
        consumed (missed crossings) or objects present early in the clip
        get absorbed into the background.
        """
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        for _ in range(10):
            self._bg_sub.apply(blurred, learningRate=0.5)
        self._prev_gray = None
        logger.info("Background model primed from median frame")

    # ------------------------------------------------------------------
    # NMS helpers
    # ------------------------------------------------------------------

    def _nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """Suppress smaller detections that overlap with a larger one (IoU > threshold)."""
        sorted_dets = sorted(detections, key=lambda d: d.area, reverse=True)
        keep: List[Detection] = []
        suppressed: set = set()
        for i, det_a in enumerate(sorted_dets):
            if i in suppressed:
                continue
            keep.append(det_a)
            for j in range(i + 1, len(sorted_dets)):
                if j not in suppressed and self._iou(det_a.bbox, sorted_dets[j].bbox) > iou_threshold:
                    suppressed.add(j)
        return keep

    @staticmethod
    def _iou(
        bbox_a: Tuple[int, int, int, int],
        bbox_b: Tuple[int, int, int, int],
    ) -> float:
        xa, ya, wa, ha = bbox_a
        xb, yb, wb, hb = bbox_b
        ix1 = max(xa, xb)
        iy1 = max(ya, yb)
        ix2 = min(xa + wa, xb + wb)
        iy2 = min(ya + ha, yb + hb)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / float(wa * ha + wb * hb - inter)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_subtractor(self):
        if self._method == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._var_threshold,
                detectShadows=self._detect_shadows,
            )
        if self._method == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=self._history,
                dist2Threshold=self._var_threshold * 2,
                detectShadows=self._detect_shadows,
            )
        raise ValueError(f"Unknown background subtraction method: {self._method!r}")
