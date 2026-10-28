"""KAN-28: baseline predictors and the predict step. Offline: SQLite built from schema.sql."""

import json
import re
import socket
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from tenacity import wait_none

from pm25_forecast import predict
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models import baselines
from pm25_forecast.models.baselines import (
    CAMS_PASSTHROUGH_MODEL_VERSION,
    PERSISTENCE_MODEL_VERSION,
    cams_passthrough,
    persistence_forecast,
    valid_times,
)
from pm25_forecast.predict import PredictResult, run
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = (Path(sql.__file__).parent / "schema.sql").read_text()
NOW = datetime(2026, 10, 28, 12, 12, tzinfo=UTC)
ISSUE = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
H = timedelta(hours=1)
DB_PASSWORD = "-".join(["dummy", "pw", "xyz"])  # assembled so no credential literal is in source
DB_URL = f"postgresql://user:{DB_PASSWORD}@127.0.0.1:1/db"


class FakeDb:
    """SQLite (built from the real schema.sql) behaving like a psycopg connection factory."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(SCHEMA.replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP"))

    def __call__(self, url):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.con.commit()
        else:
            self.con.rollback()

    def cursor(self):
        return _Cursor(self.con)

    def execute(self, statement, params=()):
        params = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in params.items()}
        return self.con.execute(re.sub(r"%\((\w+)\)s", r":\1", statement), params)

    def predictions(self):
        return self.con.execute(
            "SELECT station_id, issue_time, valid_time, model_version, run_id, pm25_pred "
            "FROM predictions WHERE model_version != ? ORDER BY 1, 2, 4, 3",
            (CAMS_MODEL_VERSION,),
        ).fetchall()


class _Cursor:
    def __init__(self, con):
        self._cur = con.cursor()
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()

    def executemany(self, statement, rows):
        rows = [
            {k: v.isoformat() if isinstance(v, datetime) else v for k, v in r.items()} for r in rows
        ]
        self._cur.executemany(re.sub(r"%\((\w+)\)s", r":\1", statement), rows)
        self.rowcount = self._cur.rowcount


def load_inputs(fake: FakeDb, *, observations=True, cams=True) -> None:
    data = json.loads((FIXTURES / "predict_inputs.json").read_text())
    if observations:
        for o in data["observations"]:
            fake.con.execute(
                "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, ?, 'test')",
                (o["station_id"], o["ts"], o["pm25"]),
            )
    if cams:
        for c in data["cams_vintages"]:
            fake.con.execute(
                "INSERT INTO predictions (station_id, issue_time, valid_time, model_version,"
                " run_id, pm25_pred) VALUES (?, ?, ?, ?, 'collector', ?)",
                (
                    c["station_id"],
                    c["issue_time"],
                    c["valid_time"],
                    CAMS_MODEL_VERSION,
                    c["pm25_pred"],
                ),
            )
    fake.con.commit()


@pytest.fixture
def fake_db():
    fake = FakeDb()
    load_inputs(fake)
    return fake


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(db, "RETRY_WAIT", wait_none())


def settings(**kw):
    values = {"database_url": DB_URL, "station_locations": "59t=13.7,100.5;61t=13.8,100.6"}
    return Settings(_env_file=None, **{**values, **kw})


def do_run(fake, now=NOW, **kw) -> PredictResult:
    return run(settings(**kw), connect=fake, clock=lambda: now)


# --- pure predictors -------------------------------------------------------------------------


def test_horizon_arithmetic_is_utc():
    assert baselines.HORIZONS_H == (6, 12, 24)
    assert valid_times(ISSUE) == {
        6: datetime(2026, 10, 28, 18, 0, tzinfo=UTC),
        12: datetime(2026, 10, 29, 0, 0, tzinfo=UTC),
        24: datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
    }
    bangkok = timezone(timedelta(hours=7))
    vt = valid_times(ISSUE.astimezone(bangkok))
    assert all(v.utcoffset() == timedelta(0) for v in vt.values())
    assert vt[24] == datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


def test_persistence_uses_latest_observation_at_or_before_issue():
    obs = [
        (ISSUE - 2 * H, 10.0),
        (ISSUE, 12.0),
        (ISSUE - H, 11.0),
        (ISSUE + H, 99.0),  # after the issue time: must not leak
    ]
    forecasts = persistence_forecast(obs, ISSUE)
    assert [f.horizon_h for f in forecasts] == [6, 12, 24]
    assert {f.pm25_pred for f in forecasts} == {12.0}
    assert forecasts[-1].valid_time == ISSUE + 24 * H


def test_persistence_skips_stale_or_missing_observations():
    assert persistence_forecast([], ISSUE) == []
    assert persistence_forecast([(ISSUE - 4 * H, 20.0)], ISSUE) == []
    assert persistence_forecast([(ISSUE - H, None)], ISSUE) == []
    assert persistence_forecast([(ISSUE - H, float("nan"))], ISSUE) == []
    assert persistence_forecast([(ISSUE - 3 * H, 20.0)], ISSUE)[0].pm25_pred == 20.0


def test_cams_passthrough_uses_latest_vintage_not_after_run():
    v = ISSUE + 6 * H
    rows = [
        (ISSUE - 12 * H, v, 30.0),
        (ISSUE - 6 * H, v, 35.0),
        (ISSUE + 30 * H / 60, v, 999.0),  # issued after the run: must not be used
    ]
    found, missing = cams_passthrough(rows, ISSUE, NOW)
    assert [(f.horizon_h, f.pm25_pred) for f in found] == [(6, 35.0)]
    assert missing == [12, 24]


def test_predictors_reject_naive_datetimes():
    naive = ISSUE.replace(tzinfo=None)
    with pytest.raises(ValueError):
        valid_times(naive)
    with pytest.raises(ValueError):
        persistence_forecast([(ISSUE, 1.0)], naive)
    with pytest.raises(ValueError):
        persistence_forecast([(naive, 1.0)], ISSUE)
    with pytest.raises(ValueError):
        cams_passthrough([(ISSUE, ISSUE + H, 1.0)], ISSUE, naive)
    with pytest.raises(ValueError):
        cams_passthrough([(naive, ISSUE + H, 1.0)], ISSUE, NOW)


# --- the predict step ------------------------------------------------------------------------


def test_run_writes_both_models_for_each_horizon(fake_db):
    result = do_run(fake_db)
    assert result.failures == []
    assert result.issue_time == ISSUE
    rows = fake_db.predictions()
    got = {(r[0], r[3], r[2]): r[5] for r in rows}
    t6, t12, t24 = (ISSUE + n * H for n in (6, 12, 24))
    iso = lambda t: t.isoformat()  # noqa: E731
    assert got == {
        ("59t", PERSISTENCE_MODEL_VERSION, iso(t6)): 40.0,
        ("59t", PERSISTENCE_MODEL_VERSION, iso(t12)): 40.0,
        ("59t", PERSISTENCE_MODEL_VERSION, iso(t24)): 40.0,
        ("59t", CAMS_PASSTHROUGH_MODEL_VERSION, iso(t6)): 35.0,
        ("59t", CAMS_PASSTHROUGH_MODEL_VERSION, iso(t12)): 36.0,
    }
    assert all(r[1] == ISSUE.isoformat() and r[4] == result.run_id for r in rows)
    assert result.per_model == {PERSISTENCE_MODEL_VERSION: 3, CAMS_PASSTHROUGH_MODEL_VERSION: 2}
    assert result.rows_written == 5


def test_missing_cams_row_and_missing_observation_are_skipped_and_reported(fake_db):
    result = do_run(fake_db)
    skipped = "\n".join(result.skipped)
    assert "59t: no CAMS row for +24h" in skipped
    assert "61t: no observation" in skipped and "persistence skipped" in skipped
    assert "61t: no CAMS row for +6h" in skipped
    stations = {r[0] for r in fake_db.predictions()}
    assert stations == {"59t"}  # nothing invented for 61t
    assert result.failures == []


def test_rerun_in_the_same_hour_is_idempotent(fake_db):
    first = do_run(fake_db)
    before = fake_db.predictions()
    again = do_run(fake_db)
    later_same_hour = do_run(fake_db, now=NOW + timedelta(minutes=40))
    assert fake_db.predictions() == before
    assert first.rows_written == 5 and first.rows_ignored == 0
    assert again.rows_written == 0 and again.rows_ignored == 5
    assert later_same_hour.rows_written == 0


def test_next_hour_appends_a_new_vintage_and_keeps_the_old_one(fake_db):
    do_run(fake_db)
    old = fake_db.predictions()
    do_run(fake_db, now=NOW + H)
    new = fake_db.predictions()
    assert set(old) <= set(new) and len(new) > len(old)
    assert {r[1] for r in new} == {ISSUE.isoformat(), (ISSUE + H).isoformat()}


def test_no_inputs_at_all_is_a_failure_not_a_quiet_success():
    fake = FakeDb()
    result = do_run(fake)
    assert result.failures == ["no forecasts could be built for any station"]
    assert fake.predictions() == []


def test_naive_clock_is_rejected(fake_db):
    with pytest.raises(ValueError):
        do_run(fake_db, now=NOW.replace(tzinfo=None))


def test_naive_row_never_reaches_the_database(fake_db):
    row = {
        "station_id": "59t",
        "issue_time": ISSUE.replace(tzinfo=None),
        "valid_time": ISSUE + H,
        "model_version": "x",
        "run_id": "r",
        "pm25_pred": 1.0,
    }
    with pytest.raises(ValueError):
        sql.write_rows(fake_db, sql.INSERT_PREDICTION, [row])
    assert fake_db.predictions() == []


def test_malformed_database_rows_are_dropped_not_used(fake_db):
    fake_db.con.execute(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES ('61t', ?, 55.0, 'test')",
        ((ISSUE - 1 * H).replace(tzinfo=None).isoformat(),),  # naive ts string in the DB
    )
    result = do_run(fake_db)
    assert "61t: no observation" in "\n".join(result.skipped)


def test_config_errors(fake_db):
    with pytest.raises(predict.ConfigError):
        run(settings(database_url=None), connect=fake_db, clock=lambda: NOW)
    with pytest.raises(predict.ConfigError):
        run(settings(station_locations=""), connect=fake_db, clock=lambda: NOW)
    with pytest.raises(predict.ConfigError):
        run(settings(station_locations="bad"), connect=fake_db, clock=lambda: NOW)


# --- DB down: skip, alert, non-zero, no secrets -----------------------------------------------


def test_db_down_skips_alerts_and_exits_nonzero_without_leaking_url(caplog):
    def refuse(_url):
        raise psycopg.OperationalError(f"connection refused {DB_URL}")

    alerts: list[str] = []
    result, code = run_entrypoint(
        lambda: run(settings(), connect=refuse, clock=lambda: NOW),
        alerts.append,
        connection_errors=db.CONNECTION_ERRORS,
    )
    assert isinstance(result, RunSkipped)
    assert code == 1
    assert len(alerts) == 1 and "OperationalError" in alerts[0]
    assert DB_PASSWORD not in alerts[0] and DB_PASSWORD not in caplog.text


@pytest.fixture
def patched_main(monkeypatch):
    monkeypatch.setattr(predict, "get_settings", lambda: settings())

    def patch_run(fn):
        monkeypatch.setattr(predict, "run", fn)

    return patch_run


def test_main_exit_codes(patched_main):
    patched_main(lambda *_a, **_k: PredictResult(run_id="r", issue_time=ISSUE))
    assert predict.main() == 0

    patched_main(lambda *_a, **_k: PredictResult(run_id="r", issue_time=ISSUE, failures=["x"]))
    assert predict.main() == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    patched_main(down)
    assert predict.main() == 1

    def bad_config(*_a, **_k):
        raise predict.ConfigError("nope")

    patched_main(bad_config)
    assert predict.main() == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    patched_main(crash)
    assert predict.main() == 1
