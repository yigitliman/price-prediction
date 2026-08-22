"""Drift detection tests.

Both the reference file and the prediction database are written under tmp_path,
so nothing here touches the repo's data/ directory.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import drift


def make_reference(rng, n: int = 120) -> pd.DataFrame:
    """Block groups that look roughly like the California Housing training data."""
    return pd.DataFrame({
        "MedInc": rng.normal(4.0, 0.8, n),
        "HouseAge": rng.normal(28.0, 6.0, n),
        "AveRooms": rng.normal(5.5, 0.6, n),
        "AveBedrms": rng.normal(1.1, 0.1, n),
        "Population": rng.normal(1400.0, 200.0, n),
        "AveOccup": rng.normal(3.0, 0.4, n),
        "Latitude": rng.normal(35.5, 1.0, n),
        "Longitude": rng.normal(-119.5, 1.0, n),
    })


def make_shifted(rng, n: int = 120) -> pd.DataFrame:
    """Wealthier, newer, emptier block groups a few degrees north."""
    return pd.DataFrame({
        "MedInc": rng.normal(11.0, 0.8, n),
        "HouseAge": rng.normal(4.0, 1.0, n),
        "AveRooms": rng.normal(9.5, 0.6, n),
        "AveBedrms": rng.normal(2.4, 0.1, n),
        "Population": rng.normal(300.0, 60.0, n),
        "AveOccup": rng.normal(1.2, 0.2, n),
        "Latitude": rng.normal(40.0, 0.5, n),
        "Longitude": rng.normal(-123.5, 0.5, n),
    })


def write_current(db_path: str, frame: pd.DataFrame) -> None:
    """Persist rows the way serve.py's log_prediction would have."""
    columns = ", ".join(f"{name} REAL" for name in drift.FEATURES)
    placeholders = ", ".join("?" for _ in drift.FEATURES)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"""CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns},
                    predicted_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
        )
        conn.executemany(
            f"INSERT INTO predictions ({', '.join(drift.FEATURES)}) VALUES ({placeholders})",
            frame[drift.REFERENCE_COLS].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    database = tmp_path / "predictions.db"
    report = tmp_path / "drift_report.html"
    monkeypatch.setattr(drift, "REFERENCE_PATH", str(reference))
    monkeypatch.setattr(drift, "DB_PATH", str(database))
    monkeypatch.setattr(drift, "REPORT_PATH", str(report))
    return {"reference": reference, "database": database, "report": report}


def test_drift_detected_when_inputs_shift(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), make_shifted(rng))

    result = drift.run_drift_report()

    assert result["drift_detected"]
    assert result["current_rows"] == 120
    assert paths["report"].exists()


def test_no_drift_when_current_matches_reference(paths):
    rng = np.random.default_rng(0)
    reference = make_reference(rng)
    reference.to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), reference.copy())

    result = drift.run_drift_report()

    assert not result["drift_detected"]


def test_missing_reference_is_reported_not_raised(paths):
    result = drift.run_drift_report()
    assert result == {"drift_detected": False, "reason": "no_reference_data"}


def test_too_few_rows_is_reported_not_raised(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(str(paths["database"]), make_shifted(rng, n=5))

    result = drift.run_drift_report()

    assert result == {"drift_detected": False, "reason": "insufficient_data"}
