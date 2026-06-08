# VisionCount

Real-time object detection, tracking, and counting system for conveyor belts.
Runs on Raspberry Pi 4 (and any Linux/macOS/Windows machine) using Python and OpenCV.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Folder Structure](#folder-structure)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Installation](#software-installation)
5. [Quick Start](#quick-start)
6. [Configuration Reference](#configuration-reference)
7. [How It Works — Step by Step](#how-it-works--step-by-step)
8. [Calibration Guide](#calibration-guide)
9. [Data Logs](#data-logs)
10. [Performance Optimisation on Raspberry Pi 4](#performance-optimisation-on-raspberry-pi-4)
11. [YOLO Alternative Detector](#yolo-alternative-detector)
12. [Keyboard Controls](#keyboard-controls)
13. [Testing](#testing)
14. [Future Improvements](#future-improvements)

---

## Architecture

```
Camera → Preprocessor → Detector → Tracker → Counter → Logger
                                      ↓
                                   Display
```

| Module | Class | Responsibility |
|--------|-------|----------------|
| `src/camera/capture.py` | `CameraCapture` | USB / PiCamera, threaded frame grab |
| `src/detection/background_subtractor.py` | `BackgroundSubtractorDetector` | MOG2/KNN + contours |
| `src/detection/yolo_detector.py` | `YOLODetector` | YOLOv8/v11 alternative |
| `src/tracking/centroid_tracker.py` | `CentroidTracker` | Stable IDs across frames |
| `src/counting/line_counter.py` | `LineCounter` | Virtual line crossing logic |
| `src/data_logging/data_logger.py` | `DataLogger` | CSV + SQLite persistence |
| `src/ui/display.py` | `Display` | Annotated frame renderer |
| `utils/fps_counter.py` | `FPSCounter` | Rolling-window FPS |
| `calibration/calibrate.py` | — | Interactive tuning tool |
| `main.py` | `ConveyorCounter` | Pipeline orchestrator |

---

## Folder Structure

```
visioncount/
├── config/
│   └── settings.yaml         ← all tunable parameters
├── src/
│   ├── camera/capture.py
│   ├── detection/
│   │   ├── background_subtractor.py
│   │   └── yolo_detector.py
│   ├── tracking/centroid_tracker.py
│   ├── counting/line_counter.py
│   ├── data_logging/data_logger.py
│   └── ui/display.py
├── utils/fps_counter.py
├── calibration/calibrate.py  ← interactive parameter tuner
├── tests/
│   ├── test_centroid_tracker.py
│   ├── test_line_counter.py
│   └── test_data_logger.py
├── logs/                     ← CSV and SQLite written here
├── snapshots/                ← per-count JPEG snapshots
├── models/                   ← place YOLO .pt or .tflite here
├── main.py
└── requirements.txt
```

---

## Hardware Requirements

| Component | Recommendation |
|-----------|---------------|
| Raspberry Pi 4 | 4 GB or 8 GB RAM |
| Camera | USB webcam (UVC) **or** Raspberry Pi Camera Module v2/v3 |
| Storage | microSD ≥ 16 GB (Class 10 / A2) |
| Lighting | Consistent overhead light; avoid flickering fluorescents |
| Mount | Camera perpendicular to belt, 30–60 cm above, centred |

---

## Software Installation

### 1 — Raspberry Pi OS (Bookworm recommended)

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip libopencv-dev python3-opencv
```

### 2 — Clone and install Python dependencies

```bash
git clone https://github.com/yourorg/visioncount.git
cd visioncount
pip3 install -r requirements.txt
```

> On Raspberry Pi, `opencv-python` in pip can fail.  Use the system package
> `python3-opencv` (already installed above) and remove `opencv-python` from
> requirements.txt.

### 3 — (Optional) Pi Camera Module

```bash
sudo apt-get install -y python3-picamera2
# Then set use_picamera: true in config/settings.yaml
```

### 4 — (Optional) YOLO

```bash
pip3 install ultralytics
# Download the nano model (fastest on Pi):
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Move the downloaded model:
mv yolov8n.pt models/
```

---

## Quick Start

```bash
# Run with default config (USB camera at index 0)
python3 main.py

# Override source and method from CLI
python3 main.py --source /dev/video0 --method background_subtraction

# Run calibration tool first (recommended for new setups)
python3 calibration/calibrate.py
```

---

## Configuration Reference

All parameters live in `config/settings.yaml`.  Key sections:

### `detection`

| Key | Default | Effect |
|-----|---------|--------|
| `method` | `background_subtraction` | Switch to `yolo` for YOLO detector |
| `min_contour_area` | 2000 | Ignore blobs smaller than N px² (noise filter) |
| `max_contour_area` | 50000 | Ignore blobs larger than N px² (belt-edge filter) |
| `var_threshold` | 50 | MOG2 sensitivity — lower = more sensitive |
| `history` | 500 | Frames used to build background model |
| `blur_kernel` | 21 | Gaussian blur radius — must be odd |
| `dilate_iterations` | 2 | Morphological dilation passes — fills gaps |

### `counting`

| Key | Default | Effect |
|-----|---------|--------|
| `line_position` | 0.5 | Fraction of frame height (0 = top, 1 = bottom) |
| `line_orientation` | `horizontal` | `horizontal` or `vertical` |
| `direction` | `top_to_bottom` | Which travel direction triggers a count |
| `min_speed` | 2.0 | px/frame — prevents stationary jitter counts |

### `performance`

| Key | Default | Effect |
|-----|---------|--------|
| `frame_skip` | 0 | Process 1 in every N+1 frames |
| `resize_factor` | 1.0 | Scale frames before detection (0.5 = quarter CPU) |
| `use_threading` | true | Dedicated capture thread |

---

## How It Works — Step by Step

### 1. Camera Capture
`CameraCapture` opens the camera with OpenCV (`cv2.VideoCapture`) or
`picamera2`.  A background thread continuously reads frames into a slot;
the main loop always gets the most recent frame, never a buffered stale one.

### 2. Preprocessing
The frame is optionally downscaled (`resize_factor`) to trade resolution for
CPU headroom.  This is the single largest performance knob on Pi.

### 3. Detection — Background Subtraction
```
BGR frame
  └─ greyscale + Gaussian blur          (remove high-freq noise)
  └─ MOG2 background subtractor         (foreground mask)
  └─ threshold > 200                    (drop shadows)
  └─ erode → dilate → morphClose        (fill holes, merge split blobs)
  └─ findContours + area filter         → Detection objects
```
Each `Detection` carries a bounding box, centroid, and area.

### 4. Tracking — Centroid Tracker
On each frame the tracker:
1. Builds a distance matrix between existing tracked centroids and new
   detection centroids using `scipy.spatial.distance.cdist`.
2. Greedily assigns the closest pairs within `max_distance`.
3. Unmatched tracked objects increment their `disappeared` counter; removed
   after `max_disappeared` frames.
4. Unmatched new detections are registered with fresh integer IDs.

Each `TrackedObject` records the last 30 centroid positions (trajectory),
per-frame velocity, and a `counted` flag.

### 5. Counting — Line Counter
The counter inspects each tracked object's last two centroid positions.
A crossing is detected when:
- `prev_y < line_y ≤ curr_y`  (top-to-bottom)

The object ID is added to `_counted_ids` (a set) immediately — subsequent
frames never recount it even if the object lingers on the line.
A configurable `min_speed` guard prevents stationary false positives.

### 6. Logging
On each count event the `on_count` callback fires, which calls
`DataLogger.log()`.  Both CSV and SQLite are written with:
- ISO-8601 timestamp
- Object ID
- Running total
- Speed (px/frame)

### 7. Display
`Display.render()` draws on a **copy** of the frame:
- Green line with "COUNT LINE" label
- Orange bounding boxes (red once counted)
- Centroid dots + trajectory trail
- ID labels
- Count panel (top-right)
- FPS (top-left)

### 8. Keyboard Control Loop
```
q / ESC  quit
r        reset counter + tracker
b        rebuild background model
s        manual snapshot
p        pause / resume
+ / -    move counting line
```

---

## Calibration Guide

Run the interactive tool before deploying:

```bash
python3 calibration/calibrate.py --source 0
```

### Tuning Checklist

**Step 1 — Set counting line position**
- Click anywhere on the live frame to place the line.
- Choose a position where boxes are clearly separated (not overlapping).
- Mid-belt usually works; avoid the entry/exit edges where detections are noisy.

**Step 2 — Tune `min_contour_area`**
- Press `+` / `-` to increase / decrease.
- Raise until conveyor belt texture noise disappears.
- Lower until small boxes are detected.
- Rule of thumb: set to ~50% of the smallest box's pixel area.

**Step 3 — Tune `var_threshold`**
- Press `t` / `T` to increase / decrease.
- Increase if the belt pattern triggers false detections on an empty belt.
- Decrease if boxes are not detected.
- Start at 50; range 20–100 covers most scenarios.

**Step 4 — Tune for lighting**
- Flickering lights → increase `history` to 1000 to smooth background.
- Strong shadows → enable `detect_shadows: true` and leave `threshold=200`
  to discard shadow pixels (grey = 127).
- Overexposure → reduce camera exposure in OS camera settings.

**Step 5 — Tune for box size**
- Small boxes: decrease `min_contour_area`, increase `dilate_iterations` to 3.
- Large boxes: increase `max_contour_area`.
- Touching boxes: decrease `var_threshold` (more sensitive) and check that
  the morphological close kernel doesn't merge two separate boxes.

**Step 6 — Tune for conveyor speed**
- Fast belt: increase `max_distance` in tracking (boxes move more px/frame).
- Fast belt: decrease `max_disappeared` (objects leave the frame quickly).
- Slow belt: increase `max_disappeared` if tracking breaks temporarily.

**Step 7 — Save and validate**
- Press `s` in the calibration tool to write tuned values back to
  `config/settings.yaml`.
- Run `python3 main.py` and observe 5–10 boxes.  Reopen calibration if
  any box is missed or double-counted.

---

## Data Logs

### CSV — `logs/counts.csv`

```
timestamp,object_id,total_count,speed_px_per_frame,notes
2025-01-15T09:00:01.123,0,1,8.2,
2025-01-15T09:00:03.456,1,2,7.9,
```

### SQLite — `logs/counts.db`

Query examples:

```sql
-- Boxes per hour today
SELECT strftime('%H', timestamp) AS hour, COUNT(*) AS boxes
FROM counts
WHERE date(timestamp) = date('now')
GROUP BY hour;

-- Average speed
SELECT AVG(speed_px_per_frame) FROM counts;

-- Total for this session
SELECT MAX(total_count) FROM counts;
```

---

## Performance Optimisation on Raspberry Pi 4

| Technique | Typical saving | How to enable |
|-----------|---------------|---------------|
| Resize to 320×240 | 4× CPU reduction | `resize_factor: 0.5` |
| Skip frames | 2× CPU reduction | `frame_skip: 1` |
| Threaded capture | Removes frame-read latency | `use_threading: true` (default) |
| MOG2 over KNN | ~20% faster | `background_subtractor: MOG2` |
| Reduce history | Faster model update | `history: 200` |
| Disable snapshots | Removes disk I/O spikes | `save_snapshots: false` |
| Headless mode | Skip `cv2.imshow` | Pipe to a file or use a flag |
| Overclock Pi 4 | 10–20% faster | `arm_freq=2000` in `/boot/config.txt` |

Expected FPS on Pi 4 @ 640×480:

| Method | FPS |
|--------|-----|
| BGS (no resize) | 20–25 |
| BGS + resize 0.5 | 40–50 |
| YOLOv8n (CPU) | 3–6 |
| YOLOv8n TFLite INT8 | 8–12 |
| YOLOv8n + Coral Edge TPU | 25–30 |

---

## YOLO Alternative Detector

```yaml
# config/settings.yaml
detection:
  method: "yolo"

yolo:
  model_path: "models/yolov8n.pt"
  confidence: 0.50
  iou_threshold: 0.45
  classes: null        # null = detect everything
  device: "cpu"
```

**Export to TFLite for faster Pi inference:**

```bash
yolo export model=yolov8n.pt format=tflite int8=True
mv yolov8n_int8.tflite models/
# Then update model_path: "models/yolov8n_int8.tflite"
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `r` | Reset counter and tracker |
| `b` | Rebuild background model (after lighting change) |
| `s` | Save manual snapshot |
| `p` | Pause / resume |
| `+` | Move counting line down / right |
| `-` | Move counting line up / left |

---

## Testing

```bash
pip3 install pytest
pytest tests/ -v
```

36 tests covering:
- Centroid tracker registration, matching, disappearance, velocity, reset
- Line counter crossing logic, directions, speed guard, callbacks, reset
- Data logger CSV/SQLite output, appending, summary, disabled mode

---

## Future Improvements

### Short term
- **DeepSORT / ByteTrack** — appearance-based re-identification for
  overcrowded belts where centroid matching breaks.
- **ROI masking** — define a polygon to exclude belt edges and mounting
  hardware from detection.
- **Multi-line counting** — count zones A→B and B→C independently.

### Medium term
- **TensorFlow Lite + Edge TPU (Coral USB Accelerator)** — 25–30 FPS
  YOLO on Pi 4 with hardware acceleration.
- **RTSP / HTTP MJPEG stream** — view the annotated feed in a browser
  without a monitor.
- **REST API** — expose `/count`, `/reset`, `/snapshot` endpoints via
  FastAPI for integration with a production line MES.
- **Shift reports** — auto-export per-shift totals to CSV at configured times.

### Long term
- **YOLOv11 + custom dataset** — train on your specific box types for
  near-100% detection accuracy under varied lighting.
- **Anomaly detection** — flag boxes that are too small, too large,
  or moving in the wrong direction.
- **Multi-camera stitching** — count across a wide belt using two cameras.
