import copy
import json
import os
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DEFAULTS_PATH = os.path.join(CONFIG_DIR, "defaults.json")


def load_defaults():
    with open(DEFAULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def sensitivity_to_min_area(sensitivity):
    sensitivity = max(0, min(100, sensitivity))
    return max(20, round(3000 - sensitivity * 28))


def migrate_color_calibration(cfg):
    """Upgrade a pre-multi-profile config (single color_calibration.hsv dict)
    into the profiles-list schema, in place. Safe to call on already-migrated
    configs (no-op if "profiles" is already present)."""
    cc = cfg.get("color_calibration") or {}
    if "profiles" in cc:
        return cfg
    old_hsv = cc.get("hsv")
    if old_hsv:
        cfg["color_calibration"] = {
            "profiles": [
                {
                    "id": 1,
                    "name": "Profile 1",
                    "enabled": True,
                    "color_space": "hsv",
                    "lower_hsv": [old_hsv["l_h"], old_hsv["l_s"], old_hsv["l_v"]],
                    "upper_hsv": [old_hsv["u_h"], old_hsv["u_s"], old_hsv["u_v"]],
                    "overlay_color": "#00FF00",
                }
            ]
        }
    else:
        cfg["color_calibration"] = {"profiles": []}
    return cfg


class ConfigManager:
    """Thread-safe in-memory config, backed by whichever project is currently
    open (see config/project_manager.py). Both the processing thread (every
    frame) and Flask request threads (on slider changes) touch this."""

    def __init__(self, initial=None):
        self._lock = threading.RLock()
        self._config = initial if initial is not None else load_defaults()

    def get_all(self):
        with self._lock:
            return copy.deepcopy(self._config)

    def get(self, section):
        with self._lock:
            return copy.deepcopy(self._config.get(section))

    def replace_all(self, new_config):
        with self._lock:
            self._config = copy.deepcopy(new_config)

    def update(self, patch):
        with self._lock:
            _deep_merge(self._config, patch)
            detection_patch = patch.get("detection", {})
            filters_patch = patch.get("filters", {})
            if "sensitivity" in detection_patch and "min_area" not in filters_patch:
                self._config["filters"]["min_area"] = sensitivity_to_min_area(detection_patch["sensitivity"])
            return copy.deepcopy(self._config)

    def reset(self):
        with self._lock:
            self._config = load_defaults()
            return copy.deepcopy(self._config)

    def export_json(self):
        with self._lock:
            return json.dumps(self._config, indent=2)

    def import_json(self, raw):
        loaded = migrate_color_calibration(json.loads(raw))
        with self._lock:
            self._config = loaded
            return copy.deepcopy(self._config)


config_manager = ConfigManager()
