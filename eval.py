"""
Counting Accuracy Evaluator
===========================
Runs the full detection → tracking → counting pipeline headless (no display)
over one or more video files and reports the final count per video, optionally
compared against a ground-truth count.

Usage:
    python3 eval.py VIDEO [VIDEO ...] [--config config/settings.yaml]
                    [--method background_subtraction|yolo]
                    [--truth NAME=COUNT,NAME=COUNT,...]

Example:
    python3 eval.py ~/Downloads/box2.mp4 ~/Downloads/box3.mp4 \
        --truth box2.mp4=4,box3.mp4=20

The harness mirrors main.py exactly: same warmup, same frame-size sync, same
counting logic — so numbers here predict live behaviour.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from main import _build_detector  # noqa: E402
from src.detection.background_subtractor import estimate_background  # noqa: E402
from src.tracking.centroid_tracker import CentroidTracker  # noqa: E402
from src.counting.line_counter import LineCounter  # noqa: E402

MEDIAN_SAMPLE_FRAMES = 25
WARMUP_FRAMES = 30
WARMUP_LEARNING_RATE = 0.05


def _sample_frames(cap, n: int, limit: int, preprocess) -> list:
    """Evenly sample n preprocessed frames from the first `limit` frames
    (whole video when limit is 0)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit:
        total = min(total, limit)
    if total <= 0:
        return []
    samples = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n))
        ok, frame = cap.read()
        if ok:
            samples.append(preprocess(frame))
    return samples


def evaluate_video(video_path: str, cfg: dict, max_frames: int = 0) -> dict:
    """Run the pipeline over a single video, return stats."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    perf_cfg = cfg.get("performance", {})
    trk_cfg = cfg["tracking"]
    cnt_cfg = cfg["counting"]
    resize_factor = perf_cfg.get("resize_factor", 1.0)
    frame_skip = perf_cfg.get("frame_skip", 0)

    detector = _build_detector(cfg)
    tracker = CentroidTracker(
        max_disappeared=trk_cfg.get("max_disappeared", 30),
        max_distance=trk_cfg.get("max_distance", 120),
    )

    # Counting geometry from the ACTUAL video size (post-resize), never the
    # configured camera size — video files keep their native resolution.
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w, proc_h = int(src_w * resize_factor), int(src_h * resize_factor)

    counter = LineCounter(
        frame_width=proc_w,
        frame_height=proc_h,
        line_position=cnt_cfg.get("line_position", 0.5),
        orientation=cnt_cfg.get("line_orientation", "vertical"),
        direction=cnt_cfg.get("direction", "both"),
        min_speed=cnt_cfg.get("min_speed", 0.5),
        zone_width=cnt_cfg.get("zone_width", 0),
    )

    def _preprocess(frame):
        if resize_factor != 1.0:
            frame = cv2.resize(
                frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR
            )
        return frame

    # ── Background priming: median over frames sampled across the clip ───
    # Falls back to fast-learning warmup + rewind when the median is
    # contaminated (belt occupancy > 50%).  Mirrors main.py._warm_up.
    if hasattr(detector, "prime"):
        samples = _sample_frames(cap, MEDIAN_SAMPLE_FRAMES, max_frames, _preprocess)
        background = estimate_background(samples) if samples else None
        if background is not None:
            detector.prime(background)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            original_lr = detector.learning_rate
            detector.learning_rate = WARMUP_LEARNING_RATE
            for _ in range(WARMUP_FRAMES):
                ok, frame = cap.read()
                if not ok:
                    break
                detector.detect(_preprocess(frame))
            detector.learning_rate = original_lr
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Main loop ─────────────────────────────────────────────────────────
    frame_index = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_index >= max_frames:
            break
        frame_index += 1
        if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
            continue
        processed += 1

        proc = _preprocess(frame)
        detections, _ = detector.detect(proc)
        objects = tracker.update([(d.centroid, d.bbox) for d in detections])
        counter.update(objects)

    cap.release()
    return {
        "video": os.path.basename(video_path),
        "frames": frame_index,
        "processed": processed,
        "count": counter.count,
        "ids_created": tracker._next_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionCount accuracy evaluator")
    parser.add_argument("videos", nargs="+", help="Video file paths")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--method", choices=["background_subtraction", "yolo", "yolo_world"],
        help="Override detection method",
    )
    parser.add_argument(
        "--truth", default="",
        help="Comma-separated NAME=COUNT ground truths, e.g. box2.mp4=4,box3.mp4=20",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop each video after N frames (0 = full video)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s — %(message)s",
    )

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.method:
        cfg["detection"]["method"] = args.method

    truth = {}
    for pair in filter(None, args.truth.split(",")):
        name, _, val = pair.partition("=")
        truth[name.strip()] = int(val)

    rows = []
    for video in args.videos:
        path = os.path.abspath(os.path.expanduser(video))
        if not os.path.isfile(path):
            print(f"[skip] not found: {path}")
            continue
        rows.append(evaluate_video(path, cfg, max_frames=args.max_frames))

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\nmethod: {cfg['detection'].get('method')}")
    header = f"{'video':<22}{'frames':>7}{'IDs':>6}{'count':>7}{'truth':>7}{'error':>7}"
    print(header)
    print("-" * len(header))
    total_err = 0
    total_truth = 0
    for r in rows:
        gt = truth.get(r["video"])
        err = "" if gt is None else r["count"] - gt
        if gt is not None:
            total_err += abs(r["count"] - gt)
            total_truth += gt
        print(
            f"{r['video']:<22}{r['frames']:>7}{r['ids_created']:>6}"
            f"{r['count']:>7}{gt if gt is not None else '—':>7}{err:>7}"
        )
    if total_truth:
        acc = 100.0 * (1 - total_err / total_truth)
        print(f"\noverall accuracy: {acc:.1f}%  (total |error| {total_err} / {total_truth} true)")


if __name__ == "__main__":
    main()
