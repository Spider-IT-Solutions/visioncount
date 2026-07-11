import csv
import datetime
import json
import os


class EventLogger:
    """Appends one row/line per counting-crossing event to output/events.csv
    and/or output/events.jsonl, gated by cfg['output']['csv_logging'] /
    ['json_logging']. Shared by the Studio pipeline and generated runtime.py
    so the log format never drifts between the two."""

    def __init__(self, project_dir, output_cfg):
        output_dir = os.path.join(project_dir, output_cfg.get("output_folder", "output"))
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "events.csv")
        self.jsonl_path = os.path.join(output_dir, "events.jsonl")
        self.csv_enabled = bool(output_cfg.get("csv_logging"))
        self.json_enabled = bool(output_cfg.get("json_logging"))

    def log(self, event):
        if not (self.csv_enabled or self.json_enabled):
            return
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "line_id": event.get("line_id"),
            "track_id": event.get("track_id"),
            "label": event.get("label"),
        }
        if self.csv_enabled:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "line_id", "track_id", "label"])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self.json_enabled:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
