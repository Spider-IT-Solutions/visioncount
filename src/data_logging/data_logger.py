"""
Data Logger
===========
Thread-safe logger that writes each count event to a CSV file and/or an
SQLite database.

Both destinations are appended to on every run (timestamps differentiate
sessions).  The SQLite schema is minimal and index-friendly so that
post-processing queries (GROUP BY DATE, etc.) stay fast even after millions
of events.
"""

import csv
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_CSV_FIELDS = ["timestamp", "object_id", "total_count", "speed_px_per_frame", "notes"]

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS counts (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL,
        object_id           INTEGER NOT NULL,
        total_count         INTEGER NOT NULL,
        speed_px_per_frame  REAL,
        notes               TEXT
    )
"""

_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_ts ON counts (timestamp)"

_INSERT = """
    INSERT INTO counts (timestamp, object_id, total_count, speed_px_per_frame, notes)
    VALUES (?, ?, ?, ?, ?)
"""


class DataLogger:
    """
    Logs count events to CSV and SQLite.

    Usage::

        with DataLogger(csv_path="logs/counts.csv", db_path="logs/counts.db") as dl:
            dl.log(object_id=7, total_count=42, speed=5.3)
    """

    def __init__(
        self,
        csv_enabled: bool = True,
        sqlite_enabled: bool = True,
        csv_path: str = "logs/counts.csv",
        db_path: str = "logs/counts.db",
    ) -> None:
        self.csv_enabled = csv_enabled
        self.sqlite_enabled = sqlite_enabled
        self.csv_path = csv_path
        self.db_path = db_path

        self._lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None
        self._db_conn: Optional[sqlite3.Connection] = None

        self._open()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        object_id: int,
        total_count: int,
        speed: float = 0.0,
        notes: str = "",
    ) -> None:
        """Write one count event.  Thread-safe."""
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        with self._lock:
            if self.csv_enabled and self._csv_writer is not None:
                self._csv_writer.writerow({
                    "timestamp": timestamp,
                    "object_id": object_id,
                    "total_count": total_count,
                    "speed_px_per_frame": round(speed, 3),
                    "notes": notes,
                })

            if self.sqlite_enabled and self._db_conn is not None:
                self._db_conn.execute(
                    _INSERT, (timestamp, object_id, total_count, round(speed, 3), notes)
                )
                self._db_conn.commit()

        logger.debug(
            "Logged: id=%d  count=%d  speed=%.2f", object_id, total_count, speed
        )

    def get_summary(self) -> dict:
        """Return aggregate stats from SQLite (empty dict if SQLite disabled)."""
        if not self.sqlite_enabled or self._db_conn is None:
            return {}
        with self._lock:
            row = self._db_conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM counts"
            ).fetchone()
        return {
            "total_events": row[0],
            "first_event": row[1],
            "last_event": row[2],
        }

    def close(self) -> None:
        """Flush and close all file handles."""
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
            if self._db_conn is not None:
                self._db_conn.close()
                self._db_conn = None
        logger.info("DataLogger closed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self.csv_enabled:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            file_exists = os.path.isfile(self.csv_path)
            # line buffering (buffering=1) ensures data survives a crash
            self._csv_file = open(self.csv_path, "a", newline="", buffering=1)
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
            if not file_exists:
                self._csv_writer.writeheader()
            logger.info("CSV log: %s", self.csv_path)

        if self.sqlite_enabled:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._db_conn.execute(_CREATE_TABLE)
            self._db_conn.execute(_CREATE_IDX)
            self._db_conn.commit()
            logger.info("SQLite log: %s", self.db_path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
