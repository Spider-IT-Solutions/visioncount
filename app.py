from flask import Flask

from camera.video_stream import VideoStream
from config.config_manager import config_manager
from config.project_manager import project_manager
from pipeline.frame_bus import frame_bus
from pipeline.processor import ProcessingThread
from utils.recorder import Recorder
from web.routes import bp, init_state


def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app


def _open_startup_project():
    name = project_manager.most_recent_project()
    if name is None:
        return project_manager.create_project("Default")
    return project_manager.open_project(name)


def main():
    cfg = _open_startup_project()
    config_manager.replace_all(cfg)

    camera = VideoStream(config_manager.get("camera"))
    recorder = Recorder()
    processor = ProcessingThread(camera, frame_bus, config_manager, recorder)

    camera.start()
    processor.start()

    init_state(camera, frame_bus, config_manager, recorder, processor)

    app = create_app()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
