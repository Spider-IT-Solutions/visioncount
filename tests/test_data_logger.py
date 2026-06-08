"""Unit tests for DataLogger."""

import csv
import os
import sqlite3
import tempfile

import pytest

from src.data_logging.data_logger import DataLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_paths(tmp_path):
    return str(tmp_path / "counts.csv"), str(tmp_path / "counts.db")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestCSV:
    def test_csv_created_with_header(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=True, sqlite_enabled=False,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=1, total_count=1)

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["object_id"] == "1"
        assert rows[0]["total_count"] == "1"

    def test_csv_appends_across_sessions(self, tmp_paths):
        csv_path, db_path = tmp_paths
        for i in range(3):
            with DataLogger(csv_enabled=True, sqlite_enabled=False,
                            csv_path=csv_path, db_path=db_path) as dl:
                dl.log(object_id=i, total_count=i + 1)

        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3

    def test_speed_is_stored(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=True, sqlite_enabled=False,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=5, total_count=10, speed=7.25)

        with open(csv_path) as fh:
            row = list(csv.DictReader(fh))[0]
        assert float(row["speed_px_per_frame"]) == pytest.approx(7.25, rel=1e-3)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

class TestSQLite:
    def test_table_created(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=True,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=1, total_count=1)

        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "counts" in tables

    def test_row_written_to_sqlite(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=True,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=7, total_count=42, speed=3.5, notes="test")

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT * FROM counts").fetchone()
        conn.close()
        # row: id, timestamp, object_id, total_count, speed, notes
        assert row[2] == 7
        assert row[3] == 42
        assert row[4] == pytest.approx(3.5, rel=1e-3)
        assert row[5] == "test"

    def test_get_summary(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=True,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=1, total_count=1)
            dl.log(object_id=2, total_count=2)
            summary = dl.get_summary()

        assert summary["total_events"] == 2
        assert summary["first_event"] is not None

    def test_multiple_logs_in_one_session(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=True,
                        csv_path=csv_path, db_path=db_path) as dl:
            for i in range(20):
                dl.log(object_id=i, total_count=i + 1)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
        conn.close()
        assert count == 20


# ---------------------------------------------------------------------------
# Both disabled (no-op mode)
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_both_disabled_does_not_crash(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=False,
                        csv_path=csv_path, db_path=db_path) as dl:
            dl.log(object_id=1, total_count=1)
        # No file should exist
        assert not os.path.exists(csv_path)

    def test_summary_empty_when_sqlite_disabled(self, tmp_paths):
        csv_path, db_path = tmp_paths
        with DataLogger(csv_enabled=False, sqlite_enabled=False,
                        csv_path=csv_path, db_path=db_path) as dl:
            summary = dl.get_summary()
        assert summary == {}
