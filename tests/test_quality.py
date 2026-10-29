"""KAN-21: data quality checks. Offline: SQLite built from schema.sql (FakeDb)."""

import socket
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from tenacity import wait_none

from pm25_forecast import quality
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models.baselines import PERSISTENCE_MODEL_VERSION
from pm25_forecast.quality import QualityResult, run
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from test_predict import DB_PASSWORD, DB_URL, FakeDb

CHALLENGER = "kan15-retrospective-24h-lag1h"
LOCATIONS = "59t=13.7,100.5;60t=13.8,100.6"
STATIONS = ("59t", "60t")
NOW = datetime(2026, 10, 30, 12, 20, tzinfo=UTC)
HOUR = datetime(2026, 10, 30, 12, tzinfo=UTC)
H = timedelta(hours=1)
P = PERSISTENCE_MODEL_VERSION


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(db, "RETRY_WAIT", wait_none())


def settings(**kw):
    values = {
        "database_url": DB_URL,
        "station_locations": LOCATIONS,
        "challenger_model_version": CHALLENGER,
    }
    return Settings(_env_file=None, **{**values, **kw})


def do_run(fake, now=NOW, **kw) -> QualityResult:
    return run(settings(**kw), connect=fake, clock=lambda: now)


def add_obs(fake, station, ts, pm25, source="air4thai"):
    fake.con.execute(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (station_id, ts) DO UPDATE SET pm25 = excluded.pm25",
        (station, ts.isoformat(), pm25, source),
    )
    fake.con.commit()


def add_pred(fake, station, issue, model):
    fake.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        " pm25_pred) VALUES (?, ?, ?, ?, 'r', 30.0)",
        (station, issue.isoformat(), (issue + 24 * H).isoformat(), model),
    )
    fake.con.commit()


def seed_obs(fake, station, newest=HOUR - H, hours=25, pm25=30.0, skip=()):
    """Hourly air4thai readings ending at `newest`, going back `hours` rows."""
    for i in range(hours):
        if i not in skip:
            add_obs(fake, station, newest - i * H, pm25)


def seed(fake, stations=STATIONS, forecast_issue=HOUR):
    for station in stations:
        seed_obs(fake, station)
        for model in (P, CAMS_MODEL_VERSION, CHALLENGER):
            add_pred(fake, station, forecast_issue, model)


@pytest.fixture
def fake():
    return FakeDb()


def get(result, source, station, check):
    found = [
        c
        for c in result.checks
        if c.source == source and c.station_id == station and c.check == check
    ]
    assert len(found) == 1, (source, station, check, found)
    return found[0]


def stored(fake):
    return fake.con.execute("SELECT count(*) FROM data_quality_checks").fetchone()[0]


def test_healthy_data_is_all_ok(fake):
    seed(fake)
    result = do_run(fake)
    assert {c.status for c in result.checks} == {"ok"}
    # 2 stations x (5 air4thai checks) + 2 stations x 3 models freshness
    assert len(result.checks) == 16 and result.written == 16 and stored(fake) == 16


@pytest.mark.parametrize(
    ("newest", "status"),
    [
        (HOUR - H, "ok"),  # 1.3 h
        (HOUR - 4 * H, "warn"),  # 4.3 h
        (HOUR - 7 * H, "fail"),  # 7.3 h
    ],
)
def test_observation_freshness(fake, newest, status):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t", newest=newest)
    check = get(do_run(fake), "air4thai", "59t", "freshness")
    assert check.status == status
    assert check.value == pytest.approx((NOW - newest) / H, abs=0.01)
    assert check.threshold == (6 if status == "fail" else 3)


def test_freshness_ignores_unusable_readings(fake):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t", newest=HOUR - 8 * H)  # usable data stops 8 h ago
    for i in range(1, 6):
        add_obs(fake, "59t", HOUR - i * H, None)  # only NULLs since
    check = get(do_run(fake), "air4thai", "59t", "freshness")
    assert check.status == "fail" and check.value > 8


def test_station_with_no_observations_fails_freshness_without_a_value(fake):
    seed(fake, stations=("60t",))
    result = do_run(fake)
    fresh = get(result, "air4thai", "59t", "freshness")
    assert fresh.status == "fail" and fresh.value is None
    assert get(result, "air4thai", "59t", "expected_station").status == "warn"


@pytest.mark.parametrize(("age_h", "status"), [(5, "ok"), (30, "warn"), (50, "fail")])
def test_cams_vintage_freshness(fake, age_h, status):
    seed(fake)
    fake.con.execute(
        "DELETE FROM predictions WHERE station_id = '59t' AND model_version = ?",
        (CAMS_MODEL_VERSION,),
    )
    add_pred(fake, "59t", HOUR - age_h * H, CAMS_MODEL_VERSION)
    check = get(do_run(fake), CAMS_MODEL_VERSION, "59t", "freshness")
    assert check.status == status


@pytest.mark.parametrize("model", [P, CHALLENGER])
@pytest.mark.parametrize(("age_h", "status"), [(1, "ok"), (4, "warn"), (8, "fail")])
def test_prediction_freshness_per_served_model(fake, model, age_h, status):
    seed(fake)
    fake.con.execute(
        "DELETE FROM predictions WHERE station_id = '59t' AND model_version = ?", (model,)
    )
    add_pred(fake, "59t", HOUR - age_h * H, model)
    result = do_run(fake)
    assert get(result, model, "59t", "freshness").status == status
    assert get(result, model, "60t", "freshness").status == "ok"


def test_missing_model_forecast_fails_and_no_challenger_means_no_check(fake):
    seed(fake)
    fake.con.execute("DELETE FROM predictions WHERE model_version = ?", (P,))
    result = do_run(fake, challenger_model_version=None)
    assert get(result, P, "59t", "freshness").status == "fail"
    assert not [c for c in result.checks if c.source == CHALLENGER]


@pytest.mark.parametrize(("dropped", "status"), [(2, "ok"), (3, "warn"), (12, "warn")])
def test_missing_hours_warns_and_never_fails(fake, dropped, status):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t", skip=set(range(2, 2 + dropped)))
    check = get(do_run(fake), "air4thai", "59t", "missing_hours")
    assert check.status == status and check.value == dropped and check.threshold == 3


def test_out_of_range_readings_fail(fake):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t")
    add_obs(fake, "59t", HOUR - 2 * H, -1.0)
    add_obs(fake, "59t", HOUR - 3 * H, 1000.5)
    add_obs(fake, "59t", HOUR - 4 * H, float("inf"))
    add_obs(fake, "59t", HOUR - 5 * H, None)  # NULL is the null-rate check's business
    add_obs(fake, "59t", HOUR - 6 * H, 0.0)  # boundaries are usable
    add_obs(fake, "59t", HOUR - 7 * H, 1000.0)
    result = do_run(fake)
    check = get(result, "air4thai", "59t", "out_of_range")
    assert check.status == "fail" and check.value == 3
    assert get(result, "air4thai", "60t", "out_of_range").status == "ok"


def test_null_rate(fake):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t")
    for i in range(2, 6):  # 4 of 25 rows: 16%
        add_obs(fake, "59t", HOUR - i * H, None)
    assert get(do_run(fake), "air4thai", "59t", "null_rate").status == "ok"
    for i in range(6, 9):  # 7 of 25 rows: 28%
        add_obs(fake, "59t", HOUR - i * H, None)
    check = get(do_run(fake), "air4thai", "59t", "null_rate")
    assert check.status == "warn" and check.value == pytest.approx(0.28)


def test_expected_station_without_rows_warns_and_unexpected_one_is_still_checked(fake):
    seed(fake, stations=("60t",))
    seed_obs(fake, "other", newest=HOUR - H)  # not in STATION_LOCATIONS, but has data
    result = do_run(fake)
    missing = get(result, "air4thai", "59t", "expected_station")
    assert missing.status == "warn" and missing.value == 0
    assert get(result, "air4thai", "60t", "expected_station").status == "ok"
    assert get(result, "air4thai", "other", "out_of_range").status == "ok"


def test_openaq_is_checked_only_where_it_has_data_and_never_fails_on_age(fake):
    seed(fake)
    seed_obs(fake, "59t", newest=HOUR - 40 * H, hours=3, source="openaq")
    result = do_run(fake)
    assert get(result, "openaq", "59t", "freshness").status == "warn"
    assert not [c for c in result.checks if c.source == "openaq" and c.station_id == "60t"]
    assert not [c for c in result.checks if c.source == "openaq" and c.check == "missing_hours"]
    assert not result.failing


def test_rerun_in_the_same_hour_does_not_duplicate_and_updates_status(fake):
    seed(fake)
    do_run(fake)
    before = stored(fake)
    do_run(fake, now=NOW + timedelta(minutes=30))  # same hour
    assert stored(fake) == before == 16
    add_obs(fake, "59t", HOUR - 2 * H, -5.0)  # data changes; re-run overwrites the row
    do_run(fake)
    row = fake.con.execute(
        "SELECT status, value FROM data_quality_checks WHERE station_id = '59t' "
        "AND check_name = 'out_of_range'"
    ).fetchall()
    assert row == [("fail", 1.0)]
    do_run(fake, now=NOW + H)  # next hour adds a new history point
    assert stored(fake) == 32


def test_source_wide_row_with_null_station_is_unique_per_hour(fake):
    params = {
        "checked_at": HOUR.isoformat(),
        "source": "air4thai",
        "station_id": None,
        "check_name": "freshness",
        "status": "ok",
        "value": 1.0,
        "threshold": 3.0,
        "detail": "",
    }
    for _ in range(2):
        fake.execute(quality.UPSERT_CHECK, params)
    assert stored(fake) == 1


def test_naive_datetime_is_rejected(fake):
    seed(fake)
    with pytest.raises(ValueError, match="timezone-aware"):
        do_run(fake, now=datetime(2026, 10, 30, 12, 20))
    assert stored(fake) == 0


def test_config_errors():
    for bad in ({"station_locations": ""}, {"station_locations": "oops"}, {"database_url": None}):
        with pytest.raises(quality.ConfigError):
            run(settings(**bad), connect=FakeDb(), clock=lambda: NOW)


def test_db_down_skips_alerts_and_exits_nonzero_without_leaking_url(caplog):
    def refuse(_url):
        raise psycopg.OperationalError(f"connection refused {DB_URL}")

    alerts: list[str] = []
    result, code = run_entrypoint(
        lambda: run(settings(), connect=refuse, clock=lambda: NOW),
        alerts.append,
        connection_errors=db.CONNECTION_ERRORS,
    )
    assert isinstance(result, RunSkipped) and code == 1
    assert len(alerts) == 1 and DB_PASSWORD not in alerts[0] and DB_PASSWORD not in caplog.text


def test_one_alert_lists_fail_items_only(fake, monkeypatch, caplog):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t", skip=set(range(3, 9)))  # 6 missing hours: warn, never alerted
    add_obs(fake, "59t", HOUR - 2 * H, -1.0)  # out of range: fail
    fake.con.execute("DELETE FROM predictions WHERE station_id = '60t' AND model_version = ?", (P,))
    fake.con.commit()
    alerts: list[str] = []
    monkeypatch.setattr(quality, "get_settings", lambda: settings())
    monkeypatch.setattr(quality, "make_alert", lambda *_a, **_k: alerts.append)
    orig = quality.run
    monkeypatch.setattr(quality, "run", lambda s: orig(s, connect=fake, clock=lambda: NOW))
    assert quality.main() == 0  # failing data is alerted, not a failed run
    assert len(alerts) == 1
    text = alerts[0]
    assert "59t out_of_range" in text and f"{P}/60t freshness" in text
    assert "missing_hours" not in text and "null_rate" not in text
    assert DB_PASSWORD not in text and DB_URL not in text


def test_no_alert_when_everything_is_ok_or_warn(fake, monkeypatch):
    seed(fake, stations=("60t",))
    seed_obs(fake, "59t", skip=set(range(3, 9)))
    add_pred(fake, "59t", HOUR, P)
    add_pred(fake, "59t", HOUR, CAMS_MODEL_VERSION)
    add_pred(fake, "59t", HOUR, CHALLENGER)
    alerts: list[str] = []
    monkeypatch.setattr(quality, "get_settings", lambda: settings())
    monkeypatch.setattr(quality, "make_alert", lambda *_a, **_k: alerts.append)
    orig = quality.run
    monkeypatch.setattr(quality, "run", lambda s: orig(s, connect=fake, clock=lambda: NOW))
    assert quality.main() == 0 and alerts == []


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(quality, "get_settings", lambda: settings())

    def use(fn):
        monkeypatch.setattr(quality, "run", fn)

    use(lambda *_a, **_k: QualityResult(run_id="r", checked_at=HOUR))
    assert quality.main() == 0
    use(lambda *_a, **_k: QualityResult(run_id="r", checked_at=HOUR, failures=["x"]))
    assert quality.main() == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    use(down)
    assert quality.main() == 1

    def bad_config(*_a, **_k):
        raise quality.ConfigError("nope")

    use(bad_config)
    assert quality.main() == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    use(crash)
    assert quality.main() == 1
