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
        script = SCHEMA.replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP")
        # SQLite autoincrements only an `integer PRIMARY KEY`
        self.con.executescript(script.replace("bigserial PRIMARY KEY", "integer PRIMARY KEY"))

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
        statement = statement.replace("clock_timestamp()", "CURRENT_TIMESTAMP")
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
                "INSERT INTO observations (station_id, ts, pm25, source, ingested_at) "
                "VALUES (?, ?, ?, 'test', ?)",
                (o["station_id"], o["ts"], o["pm25"], o["ts"]),
            )
    if cams:
        for c in data["cams_vintages"]:
            fake.con.execute(
                "INSERT INTO predictions (station_id, issue_time, valid_time, model_version,"
                " run_id, pm25_pred, created_at) VALUES (?, ?, ?, ?, 'collector', ?, ?)",
                (
                    c["station_id"],
                    c["issue_time"],
                    c["valid_time"],
                    CAMS_MODEL_VERSION,
                    c["pm25_pred"],
                    (datetime.fromisoformat(c["issue_time"]) + timedelta(minutes=1)).isoformat(),
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


def test_run_writes_only_persistence_for_every_hour_1_to_24(fake_db):
    """KAN-104: persistence for +1..+24 h (skill at every hour); +6/+12/+24 stay the headline."""
    result = do_run(fake_db)
    assert result.failures == []
    assert result.issue_time == ISSUE
    rows = fake_db.predictions()
    got = {(r[0], r[3], r[2]): r[5] for r in rows}
    assert got == {
        ("59t", PERSISTENCE_MODEL_VERSION, (ISSUE + n * H).isoformat()): 40.0 for n in range(1, 25)
    }
    assert all(r[1] == ISSUE.isoformat() and r[4] == result.run_id for r in rows)
    assert result.per_model == {PERSISTENCE_MODEL_VERSION: 24}
    assert result.rows_written == 24
    assert all(r[3] != CAMS_PASSTHROUGH_MODEL_VERSION for r in rows)


def test_champion_selection_is_frozen_for_the_hour_and_fallback_is_attributed(fake_db, monkeypatch):
    class Model:
        horizons = (24,)

        def __init__(self, version):
            self.model_version = version

        def predict(self, horizon, features):
            return 21.0

    monkeypatch.setattr(predict, "load_registered_model", lambda _con, version: Model(version))
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'v1')"
    )
    do_run(fake_db, challenger_model_version="v1")
    fake_db.con.execute("UPDATE model_champion SET model_version = 'v2' WHERE slot = 'production'")
    do_run(fake_db, now=NOW + timedelta(minutes=20), challenger_model_version="v1")
    rows = fake_db.con.execute(
        "SELECT station_id, valid_time, model_version, reason FROM serving_forecasts ORDER BY 1, 2"
    ).fetchall()
    assert [(r[0], r[2], r[3]) for r in rows] == [
        ("59t", PERSISTENCE_MODEL_VERSION, "baseline_fallback"),
        ("59t", PERSISTENCE_MODEL_VERSION, "baseline_fallback"),
        ("59t", "v1", "champion"),
    ]
    assert fake_db.con.execute(
        "SELECT champion_version FROM serving_issue_decisions"
    ).fetchall() == [("v1",)]
    assert len([r for r in fake_db.predictions() if r[3] == "v1"]) == 1
    assert not any(r[3] == "v2" for r in fake_db.predictions())


def test_shadow_challenger_predictions_never_become_serving_forecasts(fake_db, monkeypatch):
    class Model:
        horizons = (24,)

        def __init__(self, version, value):
            self.model_version = version
            self.value = value

        def predict(self, horizon, features):
            return self.value

    models = {"champion-v1": Model("champion-v1", 21.0), "shadow-v2": Model("shadow-v2", 99.0)}
    monkeypatch.setattr(predict, "load_registered_model", lambda _con, version: models[version])
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'champion-v1')"
    )

    result = do_run(fake_db, challenger_model_version="shadow-v2")

    issue_predictions = [row for row in fake_db.predictions() if row[1] == ISSUE.isoformat()]
    assert {row[3] for row in issue_predictions} == {
        PERSISTENCE_MODEL_VERSION,
        "champion-v1",
        "shadow-v2",
    }
    served = fake_db.con.execute(
        "SELECT valid_time, model_version, reason FROM serving_forecasts ORDER BY valid_time"
    ).fetchall()
    assert served == [
        ((ISSUE + 6 * H).isoformat(), PERSISTENCE_MODEL_VERSION, "baseline_fallback"),
        ((ISSUE + 12 * H).isoformat(), PERSISTENCE_MODEL_VERSION, "baseline_fallback"),
        ((ISSUE + 24 * H).isoformat(), "champion-v1", "champion"),
    ]
    assert result.per_model["shadow-v2"] == sum(row[3] == "shadow-v2" for row in issue_predictions)


def test_unusable_champion_keeps_baseline_and_records_fallback(fake_db, monkeypatch):
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'bad')"
    )

    def broken(_con, _version):
        raise predict.ModelLoadError("invalid model")

    monkeypatch.setattr(predict, "load_registered_model", broken)
    result = do_run(fake_db)
    assert result.failures and "champion bad" in result.failures[0]
    assert any(row[3] == PERSISTENCE_MODEL_VERSION for row in fake_db.predictions())
    assert fake_db.con.execute(
        "SELECT DISTINCT reason FROM serving_forecasts WHERE station_id = '59t'"
    ).fetchall() == [("baseline_fallback",)]


def test_serving_attribution_uses_persisted_prediction_run(fake_db):
    target = ISSUE + timedelta(hours=6)
    fake_db.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id, "
        "pm25_pred) VALUES (?, ?, ?, ?, ?, ?)",
        ("59t", ISSUE.isoformat(), target.isoformat(), PERSISTENCE_MODEL_VERSION, "earlier", 99),
    )
    do_run(fake_db)
    row = fake_db.con.execute(
        "SELECT model_version, run_id FROM serving_forecasts WHERE station_id = ? "
        "AND valid_time = ?",
        ("59t", target.isoformat()),
    ).fetchone()
    assert row == (PERSISTENCE_MODEL_VERSION, "earlier")


def test_empty_run_does_not_freeze_the_hour(fake_db):
    fake_db.con.execute("DELETE FROM observations")
    result = do_run(fake_db)
    assert result.failures
    assert fake_db.con.execute("SELECT COUNT(*) FROM serving_issue_decisions").fetchone() == (0,)
    assert fake_db.con.execute("SELECT COUNT(*) FROM serving_forecasts").fetchone() == (0,)
    load_inputs(fake_db, cams=False)
    assert not do_run(fake_db).failures


def test_pointer_change_after_empty_run_can_be_selected_in_same_hour(fake_db, monkeypatch):
    fake_db.con.execute("DELETE FROM observations")
    assert do_run(fake_db).failures
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'v2')"
    )

    class Model:
        model_version = "v2"
        horizons = (24,)

        def predict(self, horizon, features):
            return 20.0

    monkeypatch.setattr(predict, "load_registered_model", lambda *_: Model())
    load_inputs(fake_db, cams=False)
    assert not do_run(fake_db, now=NOW + timedelta(minutes=20)).failures
    assert fake_db.con.execute(
        "SELECT champion_version FROM serving_issue_decisions"
    ).fetchone() == ("v2",)


def test_empty_retry_with_same_run_id_does_not_remove_existing_decision(fake_db):
    do_run(fake_db)
    fake_db.con.execute("DELETE FROM observations")
    do_run(fake_db)
    assert fake_db.con.execute("SELECT COUNT(*) FROM serving_issue_decisions").fetchone() == (1,)


def test_recovered_champion_does_not_rewrite_same_hour_fallback(fake_db, monkeypatch):
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'v1')"
    )

    class Model:
        model_version = "v1"
        horizons = (24,)

        def predict(self, horizon, features):
            return 21.0

    def broken(_con, _version):
        raise ValueError("malformed model")

    monkeypatch.setattr(predict, "load_registered_model", broken)
    assert do_run(fake_db).failures
    monkeypatch.setattr(predict, "load_registered_model", lambda *_: Model())
    do_run(fake_db, now=NOW + timedelta(minutes=20))
    assert fake_db.con.execute(
        "SELECT DISTINCT model_version, reason FROM serving_forecasts WHERE station_id = '59t'"
    ).fetchall() == [(PERSISTENCE_MODEL_VERSION, "baseline_fallback")]


def test_real_invalid_bundle_falls_back_without_rolling_back_baseline(fake_db):
    import hashlib

    from pm25_forecast.features import SMOOTHER_FEATURE_NAMES

    artifact = b'{"format":1,"smoother":{"beta":0.5,"horizons":"invalid"}}'
    digest = hashlib.sha256(artifact).hexdigest()
    manifest = {
        "version": "bad",
        "metrics": {},
        "training_window": {},
        "feature_list": list(SMOOTHER_FEATURE_NAMES),
        "data_sha256": "0" * 64,
        "artifact_sha256": digest,
    }
    fake_db.con.execute(
        "INSERT INTO models(model_version, sha256, artifact, manifest) VALUES (?, ?, ?, ?)",
        ("bad", digest, artifact, json.dumps(manifest)),
    )
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'bad')"
    )
    result = do_run(fake_db)
    assert result.failures and "champion bad" in result.failures[0]
    assert any(row[3] == PERSISTENCE_MODEL_VERSION for row in fake_db.predictions())


def test_champion_prediction_error_reports_role_and_keeps_baseline(fake_db, monkeypatch):
    fake_db.con.execute(
        "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'bad')"
    )

    class BrokenModel:
        model_version = "bad"
        horizons = (24,)

        def predict(self, horizon, features):
            raise ValueError("bad inputs")

    monkeypatch.setattr(predict, "load_registered_model", lambda *_: BrokenModel())
    result = do_run(fake_db)
    assert any("champion bad dropped" in error for error in result.failures)
    assert any(row[3] == PERSISTENCE_MODEL_VERSION for row in fake_db.predictions())


def test_predict_ignores_rows_ingested_after_the_issue_time(fake_db):
    fake_db.con.execute(
        "UPDATE observations SET ingested_at = ? WHERE station_id = '59t' AND ts = ?",
        ((ISSUE + timedelta(minutes=1)).isoformat(), ISSUE.isoformat()),
    )
    fake_db.con.execute(
        "UPDATE predictions SET created_at = ? WHERE station_id = '59t' "
        "AND issue_time = ? AND valid_time = ? AND model_version = ?",
        (
            (ISSUE + timedelta(minutes=1)).isoformat(),
            (ISSUE - 6 * H + timedelta(minutes=7)).isoformat(),
            (ISSUE + 6 * H).isoformat(),
            CAMS_MODEL_VERSION,
        ),
    )

    do_run(fake_db)
    rows = fake_db.predictions()
    got = {(r[3], r[2]): r[5] for r in rows if r[0] == "59t"}

    assert got[(PERSISTENCE_MODEL_VERSION, (ISSUE + 6 * H).isoformat())] == 38.0


def test_missing_observation_is_skipped_and_reported(fake_db):
    result = do_run(fake_db)
    skipped = "\n".join(result.skipped)
    assert "61t: no observation" in skipped and "persistence skipped" in skipped
    assert "CAMS" not in skipped
    stations = {r[0] for r in fake_db.predictions()}
    assert stations == {"59t"}  # nothing invented for 61t
    assert result.failures == []


def test_rerun_in_the_same_hour_is_idempotent(fake_db):
    first = do_run(fake_db)
    before = fake_db.predictions()
    again = do_run(fake_db)
    later_same_hour = do_run(fake_db, now=NOW + timedelta(minutes=40))
    assert fake_db.predictions() == before
    assert first.rows_written == 24 and first.rows_ignored == 0
    assert again.rows_written == 0 and again.rows_ignored == 24
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


def test_refetching_unchanged_hours_keeps_them_visible_to_predict(fake_db):
    """KAN-74 regression: the collector re-upserts the last 6 h every run, just before Predict."""
    rows = fake_db.con.execute(
        "SELECT station_id, ts, pm25, source FROM observations WHERE ts > ?",
        ((ISSUE - 6 * H).isoformat(),),
    ).fetchall()
    assert rows
    before = fake_db.con.execute("SELECT station_id, ts, ingested_at FROM observations").fetchall()
    for station_id, ts, pm25, source in rows:
        fake_db.execute(
            sql.UPSERT_OBSERVATION,
            {"station_id": station_id, "ts": ts, "pm25": pm25, "source": source},
        )
    after = fake_db.con.execute("SELECT station_id, ts, ingested_at FROM observations").fetchall()
    assert sorted(after) == sorted(before)  # unchanged value: ingested_at kept

    result = do_run(fake_db)  # issue_time floors to ISSUE; the re-upsert happened "now"
    assert result.failures == []
    assert {r[0] for r in fake_db.predictions()} == {"59t"}
