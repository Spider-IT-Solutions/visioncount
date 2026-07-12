import copy
import json
import os

from config.config_manager import BASE_DIR, load_defaults, migrate_color_calibration

PROJECTS_DIR = os.path.join(BASE_DIR, "Projects")


class ProjectManager:
    """Owns the Projects/<name>/ folders: config.json, runtime.py, labels.txt,
    snapshots/, output/. The Studio always has exactly one project 'open' —
    its config.json backs the live ConfigManager."""

    def __init__(self):
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        self.current_name = None

    def _project_dir(self, name):
        return os.path.join(PROJECTS_DIR, name)

    def _config_path(self, name):
        return os.path.join(self._project_dir(name), "config.json")

    def list_projects(self):
        if not os.path.isdir(PROJECTS_DIR):
            return []
        return sorted(
            entry for entry in os.listdir(PROJECTS_DIR)
            if os.path.isfile(os.path.join(PROJECTS_DIR, entry, "config.json"))
        )

    def most_recent_project(self):
        names = self.list_projects()
        if not names:
            return None
        return max(names, key=lambda n: os.path.getmtime(self._config_path(n)))

    def create_project(self, name):
        name = name.strip()
        if not name:
            raise ValueError("project name required")
        project_dir = self._project_dir(name)
        if os.path.exists(project_dir):
            raise ValueError(f"project '{name}' already exists")

        os.makedirs(os.path.join(project_dir, "snapshots"))
        os.makedirs(os.path.join(project_dir, "output"))

        cfg = load_defaults()
        cfg["project"]["name"] = name
        with open(self._config_path(name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        self._write_labels(project_dir, cfg)
        self._write_runtime(project_dir, name)
        self.current_name = name
        return cfg

    def open_project(self, name):
        path = self._config_path(name)
        if not os.path.isfile(path):
            raise ValueError(f"project '{name}' not found")
        with open(path, encoding="utf-8") as f:
            cfg = migrate_color_calibration(json.load(f))
        self.current_name = name
        return cfg

    def save_current(self, cfg):
        if not self.current_name:
            raise ValueError("no project is open")
        project_dir = self._project_dir(self.current_name)
        cfg = copy.deepcopy(cfg)
        cfg["project"]["name"] = self.current_name
        with open(self._config_path(self.current_name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self._write_runtime(project_dir, self.current_name)
        return cfg

    def _write_labels(self, project_dir, cfg):
        labels_path = os.path.join(project_dir, cfg["model"].get("labels_file") or "labels.txt")
        if not os.path.exists(labels_path):
            classes = cfg.get("filters", {}).get("classes") or ["object"]
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write("\n".join(classes) + "\n")

    def _write_runtime(self, project_dir, name):
        from pipeline.runtime_template import render_runtime
        # Explicit utf-8: on Windows, plain open(path, "w") uses the system
        # locale codepage (often cp1252), which silently mis-encodes the
        # em-dashes in the generated docstring — PyInstaller's source parser
        # then fails with UnicodeDecodeError trying to read it back as UTF-8.
        with open(os.path.join(project_dir, "runtime.py"), "w", encoding="utf-8") as f:
            f.write(render_runtime(name))


project_manager = ProjectManager()
