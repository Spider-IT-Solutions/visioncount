import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

from config.config_manager import BASE_DIR
from config.project_manager import PROJECTS_DIR


def _sanitize(name):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "app"


def _write_build_log(build_dir, cmd, result, extra_note=""):
    """Always persist the FULL (untruncated) build output to build/build_log.txt,
    regardless of what the API/UI shows inline — this is what to open/attach when
    a build fails and the on-screen message isn't enough to diagnose it."""
    log_path = os.path.join(build_dir, "build_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Build attempt: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        if extra_note:
            f.write(f"Note: {extra_note}\n")
        if result is not None:
            f.write(f"Return code: {result.returncode}\n")
            f.write("--- stdout ---\n")
            f.write(result.stdout or "(empty)")
            f.write("\n--- stderr ---\n")
            f.write(result.stderr or "(empty)")
        f.write("\n")
    return log_path


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
        return {"success": False, "exe_path": None, "log_tail": f"project '{name}' not found", "log_path": None}
    if not os.path.isfile(runtime_script):
        return {
            "success": False, "exe_path": None,
            "log_tail": "runtime.py missing — save the project first", "log_path": None,
        }

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"success": False, "exe_path": None, "log_tail": f"invalid config.json: {e}", "log_path": None}

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
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="replace")
        fake_result = subprocess.CompletedProcess(cmd, -1, stdout, stderr)
        log_path = _write_build_log(build_dir, cmd, fake_result, extra_note="TIMED OUT after 900s")
        return {
            "success": False, "exe_path": None,
            "log_tail": f"build timed out after 900s\n{stderr[-3000:]}", "log_path": log_path,
        }
    except OSError as e:
        # e.g. PyInstaller/Python not actually available at sys.executable on this machine
        log_path = _write_build_log(build_dir, cmd, None, extra_note=f"Failed to launch build process: {e}")
        return {
            "success": False, "exe_path": None,
            "log_tail": f"could not launch PyInstaller: {e}\n\n"
                        f"Check that PyInstaller is installed in this Python environment:\n"
                        f"  {sys.executable} -m pip install -r requirements.txt",
            "log_path": log_path,
        }
    except Exception:
        log_path = os.path.join(build_dir, "build_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\nUnexpected error launching build:\n{traceback.format_exc()}\n")
        return {
            "success": False, "exe_path": None,
            "log_tail": f"unexpected error launching build:\n{traceback.format_exc()[-3000:]}",
            "log_path": log_path,
        }

    log_path = _write_build_log(build_dir, cmd, result)
    log_tail = (result.stdout[-4000:] + "\n" + result.stderr[-4000:]).strip()

    if result.returncode != 0:
        return {"success": False, "exe_path": None, "log_tail": log_tail, "log_path": log_path}

    produced = None
    for candidate in (sanitized, sanitized + ".exe"):
        candidate_path = os.path.join(dist_dir, candidate)
        if os.path.isfile(candidate_path):
            produced = candidate_path
            break

    if produced is None:
        return {
            "success": False, "exe_path": None,
            "log_tail": log_tail + "\n(no output binary found in dist/)", "log_path": log_path,
        }

    final_path = os.path.join(build_dir, os.path.basename(produced))
    shutil.move(produced, final_path)
    shutil.rmtree(dist_dir, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        os.chmod(final_path, 0o755)
    except OSError:
        pass

    return {"success": True, "exe_path": final_path, "log_tail": log_tail, "log_path": log_path}
