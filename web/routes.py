import time

from flask import Blueprint, Response, jsonify, render_template, request

from config.build_manager import build_project
from config.project_manager import project_manager
from detection.registry import available_backends
from utils.snapshot import save_snapshot

bp = Blueprint("dashboard", __name__)

_state = {}


def init_state(camera, frame_bus, config_manager, recorder, processor):
    _state.update(
        camera=camera,
        frame_bus=frame_bus,
        config_manager=config_manager,
        recorder=recorder,
        processor=processor,
    )


def _normalize_lines(lines):
    existing_ids = [l["id"] for l in lines if "id" in l]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    normalized = []
    for line in lines:
        line = dict(line)
        if "id" not in line:
            line["id"] = next_id
            next_id += 1
        line.setdefault("enabled", True)
        line.setdefault("forward_label", "IN")
        line.setdefault("backward_label", "OUT")
        normalized.append(line)
    return normalized


def _switch_to_project(cfg):
    _state["config_manager"].replace_all(cfg)
    _state["camera"].reconfigure(cfg["camera"])
    _state["processor"].reset_all()


@bp.route("/")
def index():
    return render_template("dashboard.html")


def _mjpeg_generator(key):
    boundary = b"--frame"
    while True:
        jpeg = _state["frame_bus"].get_jpeg(key)
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)


@bp.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator("main"), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/video_feed/mask")
def video_feed_mask():
    return Response(_mjpeg_generator("mask"), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/video_feed/hsv")
def video_feed_hsv():
    return Response(_mjpeg_generator("hsv"), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/api/config", methods=["GET", "POST"])
def api_config():
    cfg_manager = _state["config_manager"]
    if request.method == "POST":
        return jsonify(cfg_manager.update(request.get_json(force=True)))
    return jsonify(cfg_manager.get_all())


@bp.route("/api/backends")
def api_backends():
    return jsonify(available_backends())


DEFAULT_ROI_SHAPE = {"id": 1, "name": "main", "type": "rect", "enabled": True, "x": 140, "y": 80, "w": 360, "h": 320}


@bp.route("/api/roi", methods=["GET", "POST"])
def api_roi():
    """Flat single-shape view for the existing ROI editor UI. shapes[0]."""
    cfg_manager = _state["config_manager"]
    if request.method == "POST":
        patch = request.get_json(force=True)
        roi = cfg_manager.get("roi")
        shapes = roi.get("shapes") or []
        if shapes:
            shapes[0] = {**shapes[0], **patch, "type": "rect"}
        else:
            shapes = [{**DEFAULT_ROI_SHAPE, **patch}]
        cfg_manager.update({"roi": {"shapes": shapes}})
    shapes = cfg_manager.get("roi").get("shapes") or []
    shape = shapes[0] if shapes else DEFAULT_ROI_SHAPE
    return jsonify({"enabled": shape.get("enabled", True), "x": shape["x"], "y": shape["y"], "w": shape["w"], "h": shape["h"]})


@bp.route("/api/roi/shapes", methods=["GET", "POST"])
def api_roi_shapes():
    """Full multi-shape list — power endpoint for multi-ROI support."""
    cfg_manager = _state["config_manager"]
    if request.method == "POST":
        cfg_manager.update({"roi": {"shapes": request.get_json(force=True)}})
    return jsonify(cfg_manager.get("roi").get("shapes", []))


@bp.route("/api/lines", methods=["GET", "POST"])
def api_lines():
    cfg_manager = _state["config_manager"]
    if request.method == "POST":
        lines = _normalize_lines(request.get_json(force=True))
        cfg_manager.update({"counting": {"lines": lines}})
    return jsonify(cfg_manager.get("counting")["lines"])


@bp.route("/api/lines/<int:line_id>", methods=["DELETE"])
def api_delete_line(line_id):
    cfg_manager = _state["config_manager"]
    lines = [l for l in cfg_manager.get("counting")["lines"] if l["id"] != line_id]
    cfg_manager.update({"counting": {"lines": lines}})
    return jsonify(lines)


@bp.route("/api/camera/restart", methods=["POST"])
def api_camera_restart():
    _state["camera"].restart()
    return jsonify({"restarted": True})


@bp.route("/api/camera/pause", methods=["POST"])
def api_camera_pause():
    _state["camera"].pause()
    return jsonify({"paused": True})


@bp.route("/api/camera/resume", methods=["POST"])
def api_camera_resume():
    _state["camera"].resume()
    return jsonify({"paused": False})


@bp.route("/api/camera/step", methods=["POST"])
def api_camera_step():
    delta = int(request.get_json(force=True).get("delta", 1))
    _state["camera"].step(delta)
    return jsonify({"paused": True})


@bp.route("/api/camera/seek", methods=["POST"])
def api_camera_seek():
    frame = int(request.get_json(force=True).get("frame", 0))
    _state["camera"].seek(frame)
    return jsonify({"paused": True, "frame": frame})


@bp.route("/api/stats")
def api_stats():
    return jsonify(_state["frame_bus"].get_stats())


@bp.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    frame = _state["frame_bus"].get_last_frame()
    if frame is None:
        return jsonify({"error": "no frame available yet"}), 503
    filename = save_snapshot(frame)
    return jsonify({"filename": filename})


@bp.route("/api/record/start", methods=["POST"])
def api_record_start():
    camera_cfg = _state["config_manager"].get("camera")
    filename = _state["recorder"].start(camera_cfg["width"], camera_cfg["height"], camera_cfg["fps"])
    return jsonify({"filename": filename, "recording": True})


@bp.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    filename = _state["recorder"].stop()
    return jsonify({"filename": filename, "recording": False})


@bp.route("/api/config/reset", methods=["POST"])
def api_config_reset():
    return jsonify(_state["config_manager"].reset())


@bp.route("/api/config/export")
def api_config_export():
    raw = _state["config_manager"].export_json()
    return Response(
        raw,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=config_export.json"},
    )


@bp.route("/api/config/import", methods=["POST"])
def api_config_import():
    raw = request.get_data(as_text=True)
    return jsonify(_state["config_manager"].import_json(raw))


@bp.route("/api/counts/reset", methods=["POST"])
def api_counts_reset():
    _state["processor"].reset_counts()
    return jsonify({"reset": True})


@bp.route("/api/projects", methods=["GET", "POST"])
def api_projects():
    if request.method == "POST":
        name = request.get_json(force=True).get("name", "")
        try:
            cfg = project_manager.create_project(name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        _switch_to_project(cfg)
        return jsonify({"name": project_manager.current_name, "config": cfg})
    return jsonify(project_manager.list_projects())


@bp.route("/api/projects/current")
def api_projects_current():
    return jsonify({"name": project_manager.current_name})


@bp.route("/api/projects/open", methods=["POST"])
def api_projects_open():
    name = request.get_json(force=True).get("name", "")
    try:
        cfg = project_manager.open_project(name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    _switch_to_project(cfg)
    return jsonify({"name": project_manager.current_name, "config": cfg})


@bp.route("/api/projects/save", methods=["POST"])
def api_projects_save():
    cfg = _state["config_manager"].get_all()
    try:
        saved = project_manager.save_current(cfg)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"name": project_manager.current_name, "config": saved})


@bp.route("/api/projects/build", methods=["POST"])
def api_projects_build():
    if not project_manager.current_name:
        return jsonify({"error": "no project is open"}), 400
    # Save first so the build packages the currently-tuned config, not a stale one.
    project_manager.save_current(_state["config_manager"].get_all())
    result = build_project(project_manager.current_name)
    return jsonify(result)
