import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from pm25_forecast.storage import sql
from pm25_forecast.storage.guard import DatabaseUnavailableError, RunSkipped, run_predict_step

T0 = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)


@pytest.fixture
def db():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE observations (station_id, ts, pm25, source, PRIMARY KEY (station_id, ts));
        CREATE TABLE predictions (station_id, issue_time, valid_time, model_version, run_id,
            pm25_pred, PRIMARY KEY (station_id, issue_time, valid_time, model_version));
        CREATE TABLE scores (station_id, issue_time, valid_time, model_version, pm25_obs,
            abs_error, PRIMARY KEY (station_id, issue_time, valid_time, model_version));
        """
    )
    return con


def _pred(issue, pm25, run_id="r1"):
    return {
        "station_id": "s1",
        "issue_time": issue.isoformat(),
        "valid_time": (issue + timedelta(hours=1)).isoformat(),
        "model_version": "m1",
        "run_id": run_id,
        "pm25_pred": pm25,
    }


def test_observation_upsert_is_idempotent(db):
    row = {"station_id": "s1", "ts": T0.isoformat(), "pm25": 10.0, "source": "x"}
    db.execute(sql.UPSERT_OBSERVATION, row)
    db.execute(sql.UPSERT_OBSERVATION, {**row, "pm25": 12.0})
    assert db.execute("SELECT count(*), max(pm25) FROM observations").fetchone() == (1, 12.0)


def test_predictions_are_append_only_vintages(db):
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 20.0))
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 99.0, "r2"))  # re-run: ignored
    db.execute(sql.INSERT_PREDICTION, _pred(T0 + timedelta(hours=1), 25.0))  # new vintage
    rows = db.execute("SELECT pm25_pred FROM predictions ORDER BY issue_time").fetchall()
    assert rows == [(20.0,), (25.0,)]


def test_score_upsert_is_idempotent(db):
    key = {k: v for k, v in _pred(T0, 0).items() if k not in ("run_id", "pm25_pred")}
    db.execute(sql.UPSERT_SCORE, {**key, "pm25_obs": 1.0, "abs_error": 1.0})
    db.execute(sql.UPSERT_SCORE, {**key, "pm25_obs": 2.0, "abs_error": 0.5})
    assert db.execute("SELECT count(*), max(abs_error) FROM scores").fetchone() == (1, 0.5)


def test_require_utc():
    with pytest.raises(ValueError):
        sql.require_utc(datetime(2026, 1, 1))  # noqa: DTZ001
    local = datetime(2026, 1, 1, 7, tzinfo=timezone(timedelta(hours=7)))
    assert sql.require_utc(local).hour == 0


def test_db_down_skips_and_alerts():
    alerts: list[str] = []

    def step():
        raise DatabaseUnavailableError("boom")

    result = run_predict_step(step, alerts.append)
    assert isinstance(result, RunSkipped)
    assert len(alerts) == 1


def test_success_passes_through():
    assert run_predict_step(lambda: 42, lambda _m: None) == 42
