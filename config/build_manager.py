import json
import os
import shutil
import subprocess
import sys

from config.config_manager import BASE_DIR
from config.project_manager import PROJECTS_DIR


def _sanitize(name):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "app"


def build_project(name):
    """Package Projects/<name>/ into a standalone executable via PyInstaller.
    --paths BASE_DIR lets PyInstaller's static analysis follow runtime.py's
    `from camera.video_stream import ...` etc. and bundle the whole engine
    automatically — no manual source copying. Output is native to whatever OS
    this runs on (PyInstaller does not cross-compile): a Linux binary here,
    a real .exe if the identical build is run from Windows.
    """
    project_dir = os.path.join(PROJECTS_DIR, name)
    config_path = os.path.join(project_dir, "config.json")
    runtime_script = os.path.join(project_dir, "runtime.py")

    if not os.path.isfile(config_path):
        return {"success": False, "exe_path": None, "log_tail": f"project '{name}' not found"}
    if not os.path.isfile(runtime_script):
        return {"success": False, "exe_path": None, "log_tail": "runtime.py missing — save the project first"}

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"success": False, "exe_path": None, "log_tail": f"invalid config.json: {e}"}

    build_dir = os.path.join(project_dir, "build")
    dist_dir = os.path.join(build_dir, "dist")
    work_dir = os.path.join(build_dir, "_work")
    logs_dir = os.path.join(build_dir, "logs")
    output_dir = os.path.join(build_dir, "output")
    for d in (build_dir, logs_dir, output_dir):
        os.makedirs(d, exist_ok=True)
    shutil.rmtree(dist_dir, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)

    shutil.copy2(config_path, os.path.join(build_dir, "config.json"))

    labels_name = cfg.get("model", {}).get("labels_file") or "labels.txt"
    labels_src = os.path.join(project_dir, labels_name)
    if os.path.isfile(labels_src):
        shutil.copy2(labels_src, os.path.join(build_dir, os.path.basename(labels_src)))

    model_path = cfg.get("model", {}).get("path")
    if model_path:
        model_src = model_path if os.path.isabs(model_path) else os.path.join(project_dir, model_path)
        if os.path.isfile(model_src):
            shutil.copy2(model_src, os.path.join(build_dir, os.path.basename(model_src)))

    sanitized = _sanitize(name)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--name", sanitized,
        "--distpath", dist_dir, "--workpath", work_dir, "--specpath", work_dir,
        "--paths", BASE_DIR, "--collect-all", "cv2",
        runtime_script,
    ]

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        tail = ((e.stdout or "") + "\n" + (e.stderr or ""))[-4000:]
        return {"success": False, "exe_path": None, "log_tail": f"build timed out\n{tail}"}

    log_tail = (result.stdout[-4000:] + "\n" + result.stderr[-4000:]).strip()

    if result.returncode != 0:
        return {"success": False, "exe_path": None, "log_tail": log_tail}

    produced = None
    for candidate in (sanitized, sanitized + ".exe"):
        candidate_path = os.path.join(dist_dir, candidate)
        if os.path.isfile(candidate_path):
            produced = candidate_path
            break

    if produced is None:
        return {"success": False, "exe_path": None, "log_tail": log_tail + "\n(no output binary found in dist/)"}

    final_path = os.path.join(build_dir, os.path.basename(produced))
    shutil.move(produced, final_path)
    shutil.rmtree(dist_dir, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        os.chmod(final_path, 0o755)
    except OSError:
        pass

    return {"success": True, "exe_path": final_path, "log_tail": log_tail}
