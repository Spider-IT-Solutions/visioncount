# Vision Configuration Studio

Industrial object-counting vision system for Raspberry Pi 4 (and any Linux/Windows/macOS host). Two modes:

- **Calibration Mode (the Studio)** — a Flask web dashboard for tuning detection, ROI, counting lines, tracking, and preprocessing live against an image, video file, USB camera, or RTSP stream.
- **Production Mode** — each calibrated project exports a standalone `runtime.py` (and, via **Build Application**, a packaged executable) that reads only its `config.json` and runs headlessly — no Studio, no source tree required at deploy time.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000/`. On first run it auto-creates a `Default` project under `Projects/`.

## Studio workflow

1. **New Project** (or **Open Project**) — each project is its own folder: `Projects/<name>/{config.json, runtime.py, labels.txt, snapshots/, output/}`.
2. **Calibrate** — HSV color thresholds (with a pause/step/scrub frame-by-frame calibration view), ROI rectangle(s), counting line(s) with direction labels, detection sensitivity/filters (area, aspect ratio), preprocessing (blur, CLAHE, histogram equalization, sharpen, gamma, denoise, morphology), camera settings (source, resolution, FPS, rotation, flip), and tracking parameters — all applied live, no restart.
3. **Save Configuration** — writes `config.json` and regenerates `runtime.py` for the open project.
4. **Build Application** — packages the project into a standalone executable (see below).

## Project structure

```
Projects/<name>/
├── config.json      # everything: camera, color_calibration, roi, counting,
│                     # detection, model, tracking, filters, output, logging, performance
├── runtime.py        # generated standalone CLI app (dev-mode script AND PyInstaller entry point)
├── labels.txt
├── snapshots/
├── output/           # events.csv/.jsonl, report.json/.csv, annotated video/image output
└── build/            # created by "Build Application" — packaged exe + its own config/labels/logs/output
```

## Running a project standalone (no Studio, no venv)

```bash
python Projects/<name>/runtime.py --video sample.mp4      # batch: single pass, annotated output video
python Projects/<name>/runtime.py --image photo.jpg       # single frame, annotated image
python Projects/<name>/runtime.py --camera 0               # USB camera, continuous
python Projects/<name>/runtime.py --rtsp rtsp://host/live   # IP camera, continuous
python Projects/<name>/runtime.py --display                # add a debug window to any mode
```

No args → uses `camera.*` from `config.json` as-is. Every run writes `logs/runtime.log`, `output/events.csv` + `.jsonl` (per-crossing), and `output/report.json` + `.csv` (run summary).

## Build Application (standalone executable)

Studio's **Build Application** button invokes PyInstaller against `runtime.py`, producing a single-file executable in `Projects/<name>/build/` alongside its own `config.json`/`labels.txt`/`logs/`/`output/`. Copy that `build/` folder anywhere — the executable needs nothing else installed.

**PyInstaller does not cross-compile** — it builds for whatever OS it runs on. Building here on Linux produces a Linux binary. To get a real Windows `.exe` or a macOS binary, either:

- Run the identical build on that OS (`pip install -r requirements.txt`, then click Build Application / run `config/build_manager.py`), or
- Use the included GitHub Actions workflow: **Actions → "Build Application (Windows / macOS / Linux)" → Run workflow**, enter the project name, and download the per-OS artifacts once the matrix build finishes.

## Architecture

- `pipeline/vision_pipeline.py` — the shared detect → filter → ROI gate → track → count → annotate engine, used identically by the Studio (`pipeline/processor.py`) and every generated `runtime.py`.
- `detection/` — pluggable detector backends (`hsv_contour` implemented; MobileNet-SSD/YOLOv8/YOLOv11/custom TFLite are schema-ready, Phase 3).
- `tracking/centroid_tracker.py` — nearest-centroid tracker with short occlusion tolerance.
- `counting/line_counter.py` — multi-line crossing counter (works for horizontal/vertical/angled lines with no special-casing) with per-line direction labels.
- `roi/roi_manager.py` — multi-rect ROI via OR logic (polygon ROI is schema-ready, not yet evaluated).
- `camera/video_stream.py` — threaded capture for file/USB/RTSP with pause/step/seek/loop/restart support.
- `config/` — `ConfigManager` (live in-memory config), `ProjectManager` (project folder lifecycle), `build_manager.py` (PyInstaller packaging).
- `web/`, `templates/`, `static/` — the Flask dashboard.

## Requirements

See `requirements.txt` (Flask, OpenCV, NumPy, psutil, PyInstaller). Python 3.10+.
