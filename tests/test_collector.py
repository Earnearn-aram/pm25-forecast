"""Collector tests: recorded/synthetic fixtures via httpx.MockTransport, SQLite as the DB double.

No live calls and no real database. The one "DB down" test connects psycopg to a closed local
port, which fails immediately without any server.
"""

import copy
import gzip
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import pytest

from pm25_forecast import collector
from pm25_forecast.collector import (
    ConfigError,
    find_gaps,
    is_duplicate_vintage,
    make_alert,
    run,
)
from pm25_forecast.ingest.air4thai import Observation
from pm25_forecast.ingest.open_meteo import PredictionRow
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = (Path(sql.__file__).parent / "schema.sql").read_text()
NOW = datetime(2026, 10, 26, 7, 30, tzinfo=UTC)
# TST01 in the Air4Thai fixture reads 08:00 Bangkok = 01:00 UTC with pm25 17.5.
TST01_TS = datetime(2026, 10, 26, 1, 0, tzinfo=UTC)
DB_PASSWORD = "-".join(["dummy", "pw", "xyz"])  # assembled so no credential literal is in source
DB_URL = f"postgresql://user:{DB_PASSWORD}@127.0.0.1:1/db"


class FakeDb:
    """SQLite (built from the real schema.sql) behaving like a psycopg connection factory."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(SCHEMA.replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP"))
        self.urls: list[str] = []

    def __call__(self, url):
        self.urls.append(url)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.con.commit() if exc[0] is None else self.con.rollback()

    def execute(self, statement, params=()):
        params = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in params.items()}
        return self.con.execute(re.sub(r"%\((\w+)\)s", r":\1", statement), params)

    def rows(self, table):
        return self.con.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3").fetchall()


@pytest.fixture
def fake_db():
    return FakeDb()


def load(path):
    return json.loads((FIXTURES / path).read_text())


def make_http(air=None, cams=None, status=None):
    """Route by host. `status` maps a host to an error status code."""
    air = air if air is not None else load("air4thai_getNewAQI_synthetic.json")
    cams = cams if cams is not None else load("open_meteo/cams_air_quality.json")
    weather = load("open_meteo/weather_forecast.json")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if status and host in status:
            return httpx.Response(status[host], json={"error": True, "reason": "x"})
        if host == "air4thai.pcd.go.th":
            return httpx.Response(200, json=air)
        if host == "air-quality-api.open-meteo.com":
            return httpx.Response(200, json=cams)
        if host == "api.open-meteo.com":
            return httpx.Response(200, json=weather)
        raise AssertionError(f"unexpected host {host}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.calls = calls  # type: ignore[attr-defined]
    return client


def settings(tmp_path, **kw):
    values = {
        "database_url": DB_URL,
        "station_locations": "TST01=13.75,100.5",
        "raw_archive_dir": str(tmp_path / "raw"),
    }
    return Settings(_env_file=None, **{**values, **kw})


def do_run(tmp_path, fake_db, *, now=NOW, http=None, alert=None, **kw):
    return run(
        settings(tmp_path, **kw),
        connect=fake_db,
        clock=lambda: now,
        http=http or make_http(),
        alert=alert,
    )


# --- happy path and idempotence ---------------------------------------------------------------


def test_run_writes_observation_and_cams_vintage(tmp_path, fake_db):
    result = do_run(tmp_path, fake_db)
    assert result.failures == []
    assert result.observations_written == 1
    assert [r[:2] for r in fake_db.rows("observations")] == [("TST01", TST01_TS.isoformat())]
    assert fake_db.rows("observations")[0][2:4] == (17.5, "air4thai")
    preds = fake_db.rows("predictions")
    assert len(preds) == 3  # the fixture's 09:00 hour is null and is skipped
    assert {p[1] for p in preds} == {NOW.isoformat()}  # issue_time = retrieval time
    assert {p[3] for p in preds} == {"cams_global"}
    assert fake_db.urls == [DB_URL]


def test_second_run_is_idempotent_and_cams_repeat_is_skipped(tmp_path, fake_db):
    do_run(tmp_path, fake_db)
    later = do_run(tmp_path, fake_db, now=NOW + timedelta(hours=1))
    assert later.cams_duplicates_skipped == ["TST01"]
    assert later.predictions_written == 0
    assert len(fake_db.rows("observations")) == 1
    assert len(fake_db.rows("predictions")) == 3


def test_changed_cams_content_is_stored_as_new_vintage(tmp_path, fake_db):
    do_run(tmp_path, fake_db)
    changed = load("open_meteo/cams_air_quality.json")
    changed["hourly"]["pm2_5"] = [
        v + 5 if v is not None else None for v in changed["hourly"]["pm2_5"]
    ]
    later = do_run(tmp_path, fake_db, now=NOW + timedelta(hours=1), http=make_http(cams=changed))
    assert later.cams_duplicates_skipped == []
    issues = {p[1] for p in fake_db.rows("predictions")}
    assert issues == {NOW.isoformat(), (NOW + timedelta(hours=1)).isoformat()}


def test_raw_responses_are_archived_gzip(tmp_path, fake_db):
    do_run(tmp_path, fake_db)
    air = list((tmp_path / "raw" / "air4thai").glob("*.json.gz"))
    meteo = list((tmp_path / "raw" / "open_meteo").glob("*.json.gz"))
    assert len(air) == 1
    assert len(meteo) == 2  # weather + cams
    assert json.loads(gzip.decompress(meteo[0].read_bytes()))["source"] in {
        "weather_forecast",
        "cams_forecast",
    }


# --- gap check ----------------------------------------------------------------------------------


def obs(station, ts, pm25=10.0):
    return Observation(station_id=station, ts=ts, pm25=pm25)


def test_find_gaps_counts_missing_hours_only():
    last = {"A": TST01_TS, "B": TST01_TS, "C": TST01_TS}
    new = [
        obs("A", TST01_TS + timedelta(hours=1)),  # consecutive
        obs("B", TST01_TS + timedelta(hours=4)),  # 3 missing
        obs("C", TST01_TS),  # same reading again
        obs("D", TST01_TS),  # first run for this station
    ]
    gaps = find_gaps(last, new)
    assert [(g.station_id, g.missing_hours) for g in gaps] == [("B", 3)]


def test_run_flags_gap_alerts_and_does_not_invent_rows(tmp_path, fake_db):
    before = TST01_TS - timedelta(hours=3)
    fake_db.execute(
        sql.UPSERT_OBSERVATION,
        {"station_id": "TST01", "ts": before, "pm25": 9.0, "source": "air4thai"},
    )
    alerts: list[str] = []
    result = do_run(tmp_path, fake_db, alert=alerts.append)
    assert [g.missing_hours for g in result.gaps] == [2]
    assert len(alerts) == 1
    assert "TST01" in alerts[0]
    assert len(fake_db.rows("observations")) == 2  # the stored one and the new one only


def test_no_gap_when_consecutive(tmp_path, fake_db):
    fake_db.execute(
        sql.UPSERT_OBSERVATION,
        {"station_id": "TST01", "ts": TST01_TS - STEP, "pm25": 9.0, "source": "air4thai"},
    )
    assert do_run(tmp_path, fake_db).gaps == []


STEP = timedelta(hours=1)


def test_station_without_new_reading_is_flagged_stale(tmp_path, fake_db):
    result = do_run(tmp_path, fake_db, now=TST01_TS + timedelta(hours=5))
    assert result.stale_stations == ["TST01"]  # reading is 5 h old


# --- CAMS content dedupe ------------------------------------------------------------------------


def pred_rows(values: dict[int, float], issue_hour=8):
    base = datetime(2026, 10, 26, tzinfo=UTC)
    return [
        PredictionRow(
            station_id="s",
            issue_time=base + timedelta(hours=issue_hour),
            valid_time=base + timedelta(hours=h),
            model_version="cams_global",
            run_id="r",
            pm25_pred=v,
        )
        for h, v in values.items()
    ]


def prev(values: dict[int, float]):
    base = datetime(2026, 10, 26, tzinfo=UTC)
    return {base + timedelta(hours=h): v for h, v in values.items()}


def test_dedupe_rules():
    stored = prev({7: 1.0, 8: 2.0, 9: 3.0})
    assert is_duplicate_vintage(pred_rows({8: 2.0, 9: 3.0}), stored)  # window shifted, same values
    assert not is_duplicate_vintage(pred_rows({8: 2.0, 9: 3.5}), stored)  # content differs
    assert not is_duplicate_vintage(pred_rows({8: 2.0, 9: 3.0, 10: 4.0}), stored)  # new hours
    assert not is_duplicate_vintage(pred_rows({11: 2.0}), stored)  # no overlap
    assert not is_duplicate_vintage(pred_rows({8: 2.0}), {})  # nothing stored


# --- failures, validation, config ---------------------------------------------------------------


def test_air4thai_failure_is_reported_but_cams_still_stored(tmp_path, fake_db):
    http = make_http(status={"air4thai.pcd.go.th": 404})
    result = do_run(tmp_path, fake_db, http=http)
    assert result.failures == ["air4thai fetch failed: HTTPStatusError"]
    assert result.observations_written == 0
    assert result.predictions_written == 3


def test_open_meteo_failure_is_reported_but_observations_still_stored(tmp_path, fake_db):
    http = make_http(status={"air-quality-api.open-meteo.com": 400})
    result = do_run(tmp_path, fake_db, http=http)
    assert result.failures == ["open-meteo cams failed for TST01"]
    assert result.observations_written == 1
    assert fake_db.rows("predictions") == []


def test_null_pm25_and_future_readings_are_dropped_not_stored(tmp_path, fake_db):
    air = copy.deepcopy(load("air4thai_getNewAQI_synthetic.json"))
    air["stations"][0]["AQILast"]["PM25"]["value"] = ""
    result = do_run(tmp_path, fake_db, http=make_http(air=air))
    assert (result.observations_written, result.observations_dropped) == (0, 1)

    air = copy.deepcopy(load("air4thai_getNewAQI_synthetic.json"))
    air["stations"][0]["AQILast"]["date"] = "2026-10-28"
    fake_db2 = FakeDb()
    result = do_run(tmp_path, fake_db2, http=make_http(air=air))
    assert (result.observations_written, result.observations_dropped) == (0, 1)
    assert fake_db2.rows("observations") == []


def test_config_errors(tmp_path, fake_db):
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        do_run(tmp_path, fake_db, database_url=None)
    with pytest.raises(ConfigError, match="STATION_LOCATIONS is not set"):
        do_run(tmp_path, fake_db, station_locations="")
    with pytest.raises(ConfigError, match="not ID=lat,lon"):
        do_run(tmp_path, fake_db, station_locations="TST01=abc")


def test_settings_parse_station_locations():
    s = Settings(_env_file=None, station_locations="A=1.5,2.5; B=-3,4;")
    assert s.station_location_list == [("A", 1.5, 2.5), ("B", -3.0, 4.0)]


# --- DB down: skip, alert, non-zero, no secrets in the alert -----------------------------------


def test_db_down_skips_alerts_and_exits_nonzero_without_leaking_url(tmp_path):
    alerts: list[str] = []
    result, code = run_entrypoint(
        lambda: run(settings(tmp_path), connect=db.connect, clock=lambda: NOW, http=make_http()),
        alerts.append,
        connection_errors=db.CONNECTION_ERRORS,
    )
    assert isinstance(result, RunSkipped)
    assert code == 1
    assert len(alerts) == 1
    assert DB_PASSWORD not in alerts[0]


def test_connection_errors_are_the_psycopg_ones():
    assert psycopg.OperationalError in db.CONNECTION_ERRORS
    assert psycopg.InterfaceError in db.CONNECTION_ERRORS
    assert psycopg.errors.SyntaxError not in db.CONNECTION_ERRORS  # real bugs are not "DB down"


# --- main() exit codes ---------------------------------------------------------------------------


@pytest.fixture
def patched_main(monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "get_settings", lambda: settings(tmp_path))

    def patch_run(fn):
        monkeypatch.setattr(collector, "run", fn)

    return patch_run


def test_main_exit_codes(patched_main):
    patched_main(lambda *_a, **_k: collector.CollectorResult(run_id="r"))
    assert collector.main() == 0

    patched_main(lambda *_a, **_k: collector.CollectorResult(run_id="r", failures=["x"]))
    assert collector.main() == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    patched_main(down)
    assert collector.main() == 1

    def bad_config(*_a, **_k):
        raise ConfigError("nope")

    patched_main(bad_config)
    assert collector.main() == 2


# --- alert hook ----------------------------------------------------------------------------------


def test_alert_without_webhook_logs_error(caplog):
    with caplog.at_level(logging.ERROR):
        make_alert(None)("boom")
    assert "ALERT: boom" in caplog.text


def test_alert_failure_does_not_leak_webhook_url(monkeypatch):
    url = "https://hooks.example/secret-token"

    def fail(*_a, **_k):
        raise httpx.ConnectError(f"cannot reach {url}")

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(RuntimeError) as info:
        make_alert(url)("boom")
    assert "secret-token" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
