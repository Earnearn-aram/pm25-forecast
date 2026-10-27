"""Air4Thai history client and backfill tests. No live calls: transport is mocked.

`air4thai_history_recorded_02t.json` and `air4thai_history_recorded_empty.json` are real
responses recorded on 2026-10-27 (a 2-day window, and a June chunk outside the retention
window). Larger windows are generated in the same shape by `window_handler`. The DB double is the
SQLite `FakeDb` from the collector tests (built from the real schema.sql).
"""

import json
import logging
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import pytest

from pm25_forecast import backfill
from pm25_forecast.collector import ConfigError
from pm25_forecast.ingest.air4thai import (
    Air4ThaiClient,
    Air4ThaiError,
    Observation,
    history_chunks,
    parse_history_response,
    parse_history_timestamp,
)
from pm25_forecast.settings import Settings
from pm25_forecast.storage import sql
from test_collector import DB_URL, FakeDb

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 27, 7, 30, tzinfo=UTC)  # 14:30 Bangkok
WINDOW_START = date(2026, 8, 1)  # KAN-9: Air4Thai history starts here (Bangkok time)
FIRST_TS = datetime(2026, 7, 31, 17, 0, tzinfo=UTC)  # 2026-08-01 00:00 Bangkok
LAST_TS = datetime(2026, 10, 27, 7, 0, tzinfo=UTC)  # last full hour before NOW
EXPECTED_ROWS = int((LAST_TS - FIRST_TS) / timedelta(hours=1)) + 1
STATIONS = "59t=13.7,100.5;61t=13.8,100.6;02t=13.6,100.4"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def window_handler(calls, *, fail_stations=(), status=500):
    """Mimic getHistoryData.php: hourly rows from WINDOW_START, the recorded empty body before."""
    empty = (FIXTURES / "air4thai_history_recorded_empty.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        calls.append(dict(p))
        if p["stationID"] in fail_stations:
            return httpx.Response(status)
        sdate, edate = date.fromisoformat(p["sdate"]), date.fromisoformat(p["edate"])
        day, rows = max(sdate, WINDOW_START), []
        while day <= edate:
            rows += [
                {"DATETIMEDATA": f"{day} {h:02d}:00:00", "PM25": round(10 + h / 2, 1)}
                for h in range(24)
            ]
            day += timedelta(days=1)
        if not rows:
            return httpx.Response(
                200, content=empty.replace(b'"02t"', f'"{p["stationID"]}"'.encode())
            )
        body = {
            "result": "OK",
            "error": "",
            "stations": [{"stationID": p["stationID"], "data": rows}],
        }
        return httpx.Response(200, json=body)

    return handler


def make_client(handler) -> Air4ThaiClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return Air4ThaiClient(client=http, wait_max=0.0, max_attempts=2)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, database_url=DB_URL, station_locations=STATIONS, **kw)


def run_backfill(fake_db, handler, **kw):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return backfill.run(
        settings(),
        connect=fake_db,
        clock=lambda: NOW,
        http=http,
        sleep=lambda _s: None,
        client_kwargs={"wait_max": 0.0, "max_attempts": 2},
        **kw,
    )


# --- parsing ----------------------------------------------------------------------------------


def test_recorded_response_parses_to_utc() -> None:
    obs = parse_history_response(load("air4thai_history_recorded_02t.json"), "02t")
    assert len(obs) == 48
    assert obs[0] == Observation(
        station_id="02t", ts=datetime(2026, 10, 24, 17, 0, tzinfo=UTC), pm25=12.2
    )
    assert obs[-1].ts == datetime(2026, 10, 26, 16, 0, tzinfo=UTC)  # 2026-10-26 23:00 Bangkok
    assert all(o.ts.tzinfo is not None and o.source == "air4thai" for o in obs)


def test_recorded_empty_chunk_is_not_an_error() -> None:
    assert parse_history_response(load("air4thai_history_recorded_empty.json"), "02t") == []


def test_local_to_utc_across_the_day_boundary() -> None:
    # 00:00-06:59 Bangkok is the previous UTC day; 07:00 Bangkok is 00:00 UTC.
    assert parse_history_timestamp("2026-10-01 06:00:00") == datetime(2026, 9, 30, 23, tzinfo=UTC)
    assert parse_history_timestamp("2026-10-01 07:00:00") == datetime(2026, 10, 1, 0, tzinfo=UTC)
    assert parse_history_timestamp("2026-10-01 00:00") == datetime(2026, 9, 30, 17, tzinfo=UTC)
    assert parse_history_timestamp("2026-10-01T24:00:00") == datetime(2026, 10, 1, 17, tzinfo=UTC)
    for bad in ("2026-10-01", "not a date", "2026-10-01 25:00:00"):
        with pytest.raises(ValueError):
            parse_history_timestamp(bad)


def test_synthetic_tolerates_nulls_and_format_changes(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        obs = {
            o.ts: o.pm25
            for o in parse_history_response(load("air4thai_history_synthetic.json"), "59t")
        }
    sep, octo = datetime(2026, 9, 30, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)
    hour = timedelta(hours=1)
    assert obs[sep + 16 * hour] == 20.5  # 2026-09-30 23:00 Bangkok
    assert obs[sep + 23 * hour] == 21.0  # 2026-10-01 06:00 Bangkok, string number
    assert obs[octo] == 22.0  # 2026-10-01 07:00 Bangkok, older DATETIC key
    assert obs[octo + hour] is None and obs[octo + 2 * hour] is None  # null, "-"
    assert obs[octo + 3 * hour] is None  # -1 sentinel
    assert obs[octo + 4 * hour] == 25.5  # {"value": ...} block
    assert obs[octo + 5 * hour] == 26.0  # unknown fields ignored
    assert len(obs) == 8  # the OTHER station's row and 3 malformed rows are not included
    assert "3 malformed rows skipped" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [None, [], {"stations": None}, {"result": "ERROR", "error": "x", "stations": []}],
)
def test_unusable_history_payload_raises(payload) -> None:
    with pytest.raises(Air4ThaiError):
        parse_history_response(payload, "59t")


def test_history_chunks_cover_range_without_overlap() -> None:
    chunks = history_chunks(date(2026, 6, 29), date(2026, 10, 27))
    assert chunks[0][0] == date(2026, 6, 29) and chunks[-1][1] == date(2026, 10, 27)
    assert all((b - a).days + 1 <= 31 for a, b in chunks)
    assert all(nxt[0] - prev[1] == timedelta(days=1) for prev, nxt in pairwise(chunks))
    assert len(chunks) == 4
    assert history_chunks(date(2026, 10, 1), date(2026, 10, 1)) == [(date(2026, 10, 1),) * 2]
    assert history_chunks(date(2026, 10, 2), date(2026, 10, 1)) == []
    with pytest.raises(ValueError):
        history_chunks(date(2026, 10, 1), date(2026, 10, 2), days=0)


# --- client ------------------------------------------------------------------------------------


def test_fetch_history_chunks_pauses_and_skips_empty_old_chunks() -> None:
    calls, sleeps = [], []
    client = make_client(window_handler(calls))
    result = client.fetch_history(
        "59t", NOW - timedelta(days=120), NOW, pause=1.5, sleep=sleeps.append
    )
    assert [(c["sdate"], c["edate"]) for c in calls] == [
        ("2026-06-29", "2026-07-29"),
        ("2026-07-30", "2026-08-29"),
        ("2026-08-30", "2026-09-29"),
        ("2026-09-30", "2026-10-27"),
    ]
    assert all(c["param"] == "PM25" and c["type"] == "hr" for c in calls)
    assert all(c["stime"] == "00" and c["etime"] == "23" for c in calls)
    assert sleeps == [1.5, 1.5, 1.5]  # between requests, not before the first
    assert result.empty_chunks == [(date(2026, 6, 29), date(2026, 7, 29))]
    assert result.failed_chunks == [] and not result.total_failure
    assert result.observations[0].ts == FIRST_TS
    assert result.observations[-1].ts == LAST_TS  # later hours today are outside start..end
    assert len(result.observations) == EXPECTED_ROWS


def test_fetch_history_requires_aware_bounds() -> None:
    client = make_client(window_handler([]))
    with pytest.raises(ValueError):
        client.fetch_history("59t", datetime(2026, 10, 1), NOW)  # noqa: DTZ001


def test_failed_chunk_is_recorded_and_others_still_fetched() -> None:
    calls = []
    inner = window_handler(calls)

    def handler(request):
        if request.url.params["sdate"] == "2026-08-30":
            calls.append(dict(request.url.params))
            return httpx.Response(503)
        return inner(request)

    result = make_client(handler).fetch_history(
        "59t", NOW - timedelta(days=120), NOW, sleep=lambda _s: None
    )
    assert result.failed_chunks == [(date(2026, 8, 30), date(2026, 9, 29))]
    assert len(calls) == 5  # the failing chunk was retried once (max_attempts=2)
    assert "HTTPStatusError" in result.errors[0]
    assert not result.total_failure
    assert len(result.observations) == EXPECTED_ROWS - 31 * 24


def test_invalid_json_chunk_is_a_failed_chunk() -> None:
    result = make_client(lambda r: httpx.Response(200, content=b"<html>")).fetch_history(
        "59t", NOW - timedelta(days=2), NOW, sleep=lambda _s: None
    )
    assert result.total_failure and "not valid JSON" in result.errors[0]


# --- backfill run -------------------------------------------------------------------------------


def test_backfill_upserts_every_station(caplog) -> None:
    fake_db, calls = FakeDb(), []
    with caplog.at_level(logging.INFO):
        result = run_backfill(fake_db, window_handler(calls))
    assert [r.station_id for r in result.stations] == ["59t", "61t", "02t"]
    for r in result.stations:
        assert (r.rows_fetched, r.rows_upserted, r.rows_new) == (EXPECTED_ROWS,) * 3
        assert (r.earliest, r.latest) == (FIRST_TS, LAST_TS)
        assert (r.chunks, r.empty_chunks, r.failed_chunks) == (4, 1, 0)
    assert len(fake_db.rows("observations")) == 3 * EXPECTED_ROWS
    assert not result.total_failure and result.failures == []
    assert f"station 59t: {EXPECTED_ROWS} fetched" in caplog.text
    assert "4 chunks (1 empty, 0 failed)" in caplog.text
    assert fake_db.commits == 3  # one transaction per station


def test_backfill_is_idempotent_and_does_not_duplicate_collector_rows() -> None:
    fake_db = FakeDb()
    # A row the hourly collector already stored, with the same natural key as a history row.
    with fake_db as con:
        sql.write_rows(
            con,
            sql.UPSERT_OBSERVATION,
            [Observation(station_id="59t", ts=LAST_TS, pm25=99.0).model_dump()],
        )
    first = run_backfill(fake_db, window_handler([]))
    assert first.stations[0].rows_new == EXPECTED_ROWS - 1
    rows = fake_db.rows("observations")
    second = run_backfill(fake_db, window_handler([]))
    assert len(fake_db.rows("observations")) == len(rows) == 3 * EXPECTED_ROWS
    assert all(r.rows_new == 0 for r in second.stations)
    assert all(r.rows_upserted == EXPECTED_ROWS for r in second.stations)
    stored = fake_db.con.execute(
        "SELECT count(*), max(pm25) FROM observations WHERE station_id = '59t' AND ts = ?",
        (LAST_TS.isoformat(),),
    ).fetchone()
    assert stored[0] == 1 and stored[1] != 99.0  # history value replaces, never duplicates


def test_backfill_drops_null_values_and_never_stores_hollow_rows() -> None:
    fake_db = FakeDb()
    synthetic = load("air4thai_history_synthetic.json")

    def handler(request):
        return httpx.Response(200, json=synthetic)

    result = run_backfill(fake_db, handler, station_ids=["59t"], lookback_days=40)
    report = result.stations[0]
    assert (report.rows_fetched, report.rows_dropped, report.rows_upserted) == (8, 3, 5)
    assert all(row[2] is not None for row in fake_db.rows("observations"))


def test_partial_failure_keeps_other_stations() -> None:
    fake_db = FakeDb()
    result = run_backfill(fake_db, window_handler([], fail_stations={"61t"}))
    by_id = {r.station_id: r for r in result.stations}
    assert by_id["61t"].error == "every chunk failed" and by_id["61t"].rows_upserted == 0
    assert by_id["59t"].rows_upserted == by_id["02t"].rows_upserted == EXPECTED_ROWS
    assert not result.total_failure
    assert len(result.failures) == 1 and "61t" in result.failures[0]


def test_total_failure_when_no_station_has_usable_rows() -> None:
    fake_db = FakeDb()
    result = run_backfill(fake_db, window_handler([], fail_stations={"59t", "61t", "02t"}))
    assert result.total_failure and len(result.failures) == 3
    assert fake_db.rows("observations") == [] and fake_db.urls == []


def test_tls_error_is_not_retried_and_fails_the_station() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")

    result = run_backfill(FakeDb(), handler, station_ids=["59t"], lookback_days=2)
    assert result.total_failure and result.stations[0].error == "every chunk failed"
    assert len(calls) == result.stations[0].chunks  # one attempt per chunk: no TLS retries


def test_station_ids_default_to_station_locations_and_can_be_overridden() -> None:
    assert backfill.resolve_station_ids(settings()) == ["59t", "61t", "02t"]
    assert backfill.resolve_station_ids(settings(), ["bkp100t", " ", "bkp100t"]) == ["bkp100t"]
    with pytest.raises(ConfigError):
        backfill.resolve_station_ids(Settings(_env_file=None))
    with pytest.raises(ConfigError):
        backfill.resolve_station_ids(Settings(_env_file=None, station_locations="bad"))


def test_config_errors() -> None:
    with pytest.raises(ConfigError):
        backfill.run(Settings(_env_file=None, station_locations=STATIONS))
    with pytest.raises(ConfigError):
        backfill.run(settings(), lookback_days=0)


# --- entry point exit codes ---------------------------------------------------------------------


@pytest.fixture
def entry(monkeypatch):
    alerts = []
    monkeypatch.setattr(backfill, "get_settings", settings)
    monkeypatch.setattr(backfill, "make_alert", lambda _url: alerts.append)
    return alerts


def _report(sid, fetched, **kw):
    return backfill.StationReport(sid, rows_fetched=fetched, **kw)


def test_main_exit_zero_on_partial_failure_but_alerts(monkeypatch, entry) -> None:
    result = backfill.BackfillResult([_report("59t", 10), _report("61t", 0, error="x")])
    monkeypatch.setattr(backfill, "run", lambda *a, **k: result)
    assert backfill.main(["--lookback-days", "5"]) == 0
    assert len(entry) == 1 and "61t" in entry[0]


def test_main_exit_nonzero_on_total_failure(monkeypatch, entry) -> None:
    result = backfill.BackfillResult([_report("59t", 0, error="x")])
    monkeypatch.setattr(backfill, "run", lambda *a, **k: result)
    assert backfill.main([]) == backfill.EXIT_FAILED
    assert "no usable rows" in entry[-1]


def test_main_passes_cli_arguments(monkeypatch, entry) -> None:
    seen = {}

    def fake_run(_settings, **kw):
        seen.update(kw)
        return backfill.BackfillResult([_report("bkp101t", 1)])

    monkeypatch.setattr(backfill, "run", fake_run)
    assert backfill.main(["--stations", "bkp101t", "--lookback-days", "30"]) == 0
    assert seen == {"station_ids": ["bkp101t"], "lookback_days": 30}
    assert entry == []


def test_main_config_error_exit_two(monkeypatch, entry) -> None:
    monkeypatch.setattr(backfill, "get_settings", lambda: Settings(_env_file=None))
    assert backfill.main([]) == backfill.EXIT_CONFIG
    assert "configuration error" in entry[-1]


def test_main_db_down_is_skipped_nonzero(monkeypatch, entry) -> None:
    import psycopg

    def fake_run(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(backfill, "run", fake_run)
    assert backfill.main([]) == 1
    assert "database unreachable" in entry[-1]
