import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pm25_forecast import quality
from pm25_forecast.storage import sql
from pm25_forecast.storage.guard import (
    DatabaseUnavailableError,
    RunSkipped,
    run_entrypoint,
    run_predict_step,
)

T0 = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)
SCHEMA = (Path(sql.__file__).parent / "schema.sql").read_text()


def _tables(ddl: str) -> dict[str, str]:
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", ddl, re.S)
    }


def _columns(body: str) -> list[tuple[str, str]]:
    cols = []
    for line in body.splitlines():
        m = re.match(r"\s+(\w+)\s+([a-z]+(?: precision)?)", line)
        if m and m.group(1) != "PRIMARY":
            cols.append((m.group(1), m.group(2)))
    return cols


def _primary_key(body: str) -> tuple[str, ...]:
    m = re.search(r"PRIMARY KEY \(([^)]*)\)", body)
    if m:
        return tuple(c.strip() for c in m.group(1).split(","))
    inline = [c for c, _ in _columns(body) if re.search(rf"\b{c}\s+\w+\s+PRIMARY KEY", body)]
    return tuple(inline)


class _PyformatConnection:
    """SQLite connection that accepts psycopg-style `%(name)s` placeholders (offline tests)."""

    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def execute(self, statement, params=()):
        return self._con.execute(re.sub(r"%\((\w+)\)s", r":\1", statement), params)

    def cursor(self):
        return _Cursor(self._con)


class _Cursor:
    """Context-manager cursor with psycopg-style `executemany` (rowcount totals the batch)."""

    def __init__(self, con: sqlite3.Connection):
        self._cur = con.cursor()
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()

    def executemany(self, statement, rows):
        self._cur.executemany(re.sub(r"%\((\w+)\)s", r":\1", statement), rows)
        self.rowcount = self._cur.rowcount


@pytest.fixture
def db():
    """In-memory SQLite built from the real schema.sql (no hand-written DDL to drift)."""
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP"))
    return _PyformatConnection(con)


def _pred(issue, pm25, run_id="r1", model_version="m1", valid=None):
    return {
        "station_id": "s1",
        "issue_time": issue.isoformat(),
        "valid_time": (valid or issue + timedelta(hours=1)).isoformat(),
        "model_version": model_version,
        "run_id": run_id,
        "pm25_pred": pm25,
    }


def test_observation_upsert_is_idempotent(db):
    row = {"station_id": "s1", "ts": T0.isoformat(), "pm25": 10.0, "source": "x"}
    db.execute(sql.UPSERT_OBSERVATION, row)
    db.execute(sql.UPSERT_OBSERVATION, {**row, "pm25": 12.0})
    assert db.execute("SELECT count(*), max(pm25) FROM observations").fetchone() == (1, 12.0)


def test_observation_upsert_keeps_pm25_when_new_value_is_null(db):
    row = {"station_id": "s1", "ts": T0.isoformat(), "pm25": 10.0, "source": "x"}
    db.execute(sql.UPSERT_OBSERVATION, row)
    db.execute("UPDATE observations SET ingested_at = '2000-01-01 00:00:00'")
    db.execute(sql.UPSERT_OBSERVATION, {**row, "pm25": None, "source": "y"})
    pm25, source, ingested = db.execute(
        "SELECT pm25, source, ingested_at FROM observations"
    ).fetchone()
    assert (pm25, source) == (10.0, "y")
    assert ingested != "2000-01-01 00:00:00"


def test_predictions_are_append_only_vintages(db):
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 20.0))
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 99.0, "r2"))  # re-run: ignored
    db.execute(sql.INSERT_PREDICTION, _pred(T0 + timedelta(hours=1), 25.0))  # new vintage
    rows = db.execute("SELECT pm25_pred FROM predictions ORDER BY issue_time").fetchall()
    assert rows == [(20.0,), (25.0,)]


def test_same_valid_time_different_issue_time_gives_two_rows(db):
    valid = T0 + timedelta(hours=5)
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 20.0, valid=valid))
    db.execute(sql.INSERT_PREDICTION, _pred(T0 + timedelta(hours=1), 21.0, valid=valid))
    assert db.execute("SELECT count(*) FROM predictions").fetchone() == (2,)


def test_different_model_version_gives_two_rows(db):
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 20.0, model_version="m1"))
    db.execute(sql.INSERT_PREDICTION, _pred(T0, 21.0, model_version="m2"))
    assert db.execute("SELECT count(*) FROM predictions").fetchone() == (2,)


def test_identical_key_twice_gives_one_row_first_wins(db):
    first = db.execute(sql.INSERT_PREDICTION, _pred(T0, 20.0)).rowcount
    second = db.execute(sql.INSERT_PREDICTION, _pred(T0, 99.0, "r2")).rowcount
    assert (first, second) == (1, 0)
    assert db.execute("SELECT count(*), max(pm25_pred) FROM predictions").fetchone() == (1, 20.0)


def test_schema_primary_keys_match_conflict_targets():
    tables = _tables(SCHEMA)
    assert {_primary_key(b) for b in tables.values()} >= {("station_id", "ts")}
    expected = {
        "observations": ("station_id", "ts"),
        "features": ("station_id", "ts", "feature_version"),
        "predictions": ("station_id", "issue_time", "valid_time", "model_version"),
        "scores": ("station_id", "issue_time", "valid_time", "model_version"),
        "models": ("model_version",),
        "model_champion": ("slot",),
        "data_quality_checks": ("checked_at", "source", "station_id", "check_name"),
        "drift_checks": ("checked_at", "model_version", "feature"),
        "drift_reference": ("model_version", "feature"),
        "model_events": ("id",),
    }
    assert {t: _primary_key(b) for t, b in tables.items()} == expected
    for table, stmt in (
        ("observations", sql.UPSERT_OBSERVATION),
        ("predictions", sql.INSERT_PREDICTION),
        ("scores", sql.UPSERT_SCORE),
        ("data_quality_checks", quality.UPSERT_CHECK),
    ):
        target = re.search(r"ON CONFLICT \(([^)]*)\)", stmt)
        assert target
        assert tuple(c.strip() for c in target.group(1).split(",")) == expected[table]


def test_schema_time_columns_are_timestamptz():
    time_cols = [
        (t, c, typ)
        for t, body in _tables(SCHEMA).items()
        for c, typ in _columns(body)
        if c in {"ts", "created_at", "scored_at", "ingested_at"} or c.endswith("_time")
    ]
    assert len(time_cols) >= 10
    assert [x for x in time_cols if x[2] != "timestamptz"] == []


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


def test_write_rows_rejects_naive_before_executing(db):
    row = {"station_id": "s1", "ts": datetime(2026, 1, 1), "pm25": 1.0, "source": "x"}  # noqa: DTZ001
    with pytest.raises(ValueError):
        sql.write_rows(db, sql.UPSERT_OBSERVATION, [row], adapt=lambda d: d.isoformat())
    assert db.execute("SELECT count(*) FROM observations").fetchone() == (0,)


def test_write_rows_normalises_to_utc_and_logs_ignored(db, caplog):
    local = datetime(2026, 1, 1, 7, tzinfo=timezone(timedelta(hours=7)))
    row = {**_pred(T0, 1.0), "issue_time": local, "valid_time": local + timedelta(hours=1)}
    with caplog.at_level(logging.INFO, logger="pm25_forecast.storage.sql"):
        n = sql.write_rows(db, sql.INSERT_PREDICTION, [row, row], adapt=lambda d: d.isoformat())
    assert n == 1
    assert "1 ignored" in caplog.text
    assert db.execute("SELECT issue_time FROM predictions").fetchone() == (
        "2026-01-01T00:00:00+00:00",
    )


def test_entrypoint_maps_connection_errors_and_exits_nonzero():
    alerts: list[str] = []

    def step():
        raise ConnectionRefusedError("db down")

    result, code = run_entrypoint(step, alerts.append)
    assert isinstance(result, RunSkipped)
    assert code == 1
    assert len(alerts) == 1


def test_entrypoint_custom_error_tuple_and_unrelated_errors_propagate():
    class DriverError(Exception):
        pass

    def driver_down():
        raise DriverError("x")

    result, code = run_entrypoint(driver_down, lambda _m: None, connection_errors=(DriverError,))
    assert isinstance(result, RunSkipped)
    assert code == 1

    def bug():
        raise KeyError("bug")

    with pytest.raises(KeyError):
        run_entrypoint(bug, lambda _m: None)


def test_entrypoint_success_exit_zero():
    assert run_entrypoint(lambda: 42, lambda _m: None) == (42, 0)


def test_alert_failure_is_logged_and_run_still_skipped(caplog):
    def bad_alert(_m):
        raise RuntimeError("webhook down")

    def step():
        raise DatabaseUnavailableError("boom")

    result, code = run_entrypoint(step, bad_alert)
    assert isinstance(result, RunSkipped)
    assert code == 1
    assert "alert failed" in caplog.text
