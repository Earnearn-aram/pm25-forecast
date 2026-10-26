"""Open-Meteo client tests. Fixtures are synthetic; no live calls (httpx.MockTransport)."""

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from pm25_forecast.ingest.open_meteo import (
    OpenMeteoClient,
    OpenMeteoError,
    OpenMeteoSchemaError,
    RawResponse,
    parse_hourly,
    to_snapshot_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "open_meteo"
NOW = datetime(2026, 10, 26, 12, 30, tzinfo=UTC)
BKK = (13.75, 100.5)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_client(handler, **kwargs) -> OpenMeteoClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenMeteoClient(http, retry_wait=wait_none(), now=lambda: NOW, **kwargs)


def json_handler(name: str, calls: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(200, json=load(name))

    return handler


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_fixtures_are_marked_synthetic(path: Path) -> None:
    assert json.loads(path.read_text())["_synthetic"] is True


def test_weather_forecast_parses_utc_rows_and_sets_issue_time() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("weather_forecast.json", calls))
    fc = client.weather_forecast(*BKK, ["temperature_2m", "relative_humidity_2m"])
    assert len(fc.rows) == 4
    assert all(
        r.valid_time.tzinfo is not None and r.valid_time.utcoffset() == timedelta(0)
        for r in fc.rows
    )
    assert fc.rows[0].values == {"temperature_2m": 28.1, "relative_humidity_2m": 80.0}
    assert fc.issue_time == NOW and fc.retrieved_at == NOW
    query = calls[0].url.params
    assert query["timezone"] == "UTC" and query["timeformat"] == "unixtime"
    assert query["hourly"] == "temperature_2m,relative_humidity_2m"


def test_cams_forecast_uses_global_domain_and_snapshot_rows() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("cams_air_quality.json", calls))
    fc = client.cams_forecast(*BKK)
    assert calls[0].url.params["domains"] == "cams_global"
    rows = to_snapshot_rows(fc, "bkk-test")
    assert [r.value for r in rows] == [31.2, 33.0, 35.5]  # the null hour is skipped
    assert {r.forecast_issue_time for r in rows} == {NOW}
    assert all(r.valid_time.tzinfo is not None for r in rows)
    with pytest.raises(Exception, match="frozen"):
        rows[0].value = 1.0  # type: ignore[misc]


def test_snapshots_from_two_runs_keep_distinct_vintages() -> None:
    first = make_client(json_handler("cams_air_quality.json")).cams_forecast(*BKK)
    later = OpenMeteoClient(
        httpx.Client(transport=httpx.MockTransport(json_handler("cams_air_quality.json"))),
        now=lambda: NOW + timedelta(hours=1),
    ).cams_forecast(*BKK)
    keys = {(r.forecast_issue_time, r.valid_time) for r in to_snapshot_rows(first, "x")}
    keys |= {(r.forecast_issue_time, r.valid_time) for r in to_snapshot_rows(later, "x")}
    assert len(keys) == 6  # same valid times, two issue times: nothing overwritten


def test_historical_weather_has_no_issue_time() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("historical_weather.json", calls))
    fc = client.historical_weather(*BKK, date(2026, 1, 1), date(2026, 1, 2), ["temperature_2m"])
    assert fc.issue_time is None
    assert calls[0].url.params["start_date"] == "2026-01-01"
    with pytest.raises(ValueError, match="issue_time"):
        to_snapshot_rows(fc, "x", "temperature_2m")


def test_previous_runs_hits_previous_runs_host() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("historical_weather.json", calls))
    client.previous_runs(*BKK, date(2026, 1, 1), date(2026, 1, 2), ["temperature_2m_previous_day1"])
    assert calls[0].url.host == "previous-runs-api.open-meteo.com"


def test_malformed_payload_is_tolerated(caplog: pytest.LogCaptureFixture) -> None:
    payload = load("malformed.json")
    with caplog.at_level(logging.WARNING):
        rows, missing = parse_hourly(payload, ["temperature_2m", "wind_speed_10m", "pm2_5"])
    assert len(rows) == 3  # unparseable time dropped
    assert [r.values["temperature_2m"] for r in rows] == [28.0, None, None]  # sentinel, text
    assert missing == ["wind_speed_10m"]  # renamed upstream
    assert [r.values["pm2_5"] for r in rows] == [10.0, None, None]  # short column padded
    assert "missing" in caplog.text


def test_iso_naive_times_are_treated_as_utc() -> None:
    rows, _ = parse_hourly({"hourly": {"time": ["2026-10-26T05:00"], "a": [1]}}, ["a"])
    assert rows[0].valid_time == datetime(2026, 10, 26, 5, tzinfo=UTC)


@pytest.mark.parametrize(
    "payload", [[], {"hourly": []}, {"hourly": {"temperature_2m": [1]}}, {"hourly": {"time": "x"}}]
)
def test_bad_shapes_raise_schema_error(payload) -> None:
    with pytest.raises(OpenMeteoSchemaError):
        parse_hourly(payload, ["temperature_2m"])


def test_api_error_body_raises_without_retry() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json=load("error_response.json"))

    with pytest.raises(OpenMeteoError, match="HTTP 400"):
        make_client(handler).weather_forecast(*BKK)
    assert len(calls) == 1


def test_retries_on_5xx_and_timeouts_then_succeeds() -> None:
    outcomes = iter([503, "timeout", 429, 200])
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        outcome = next(outcomes)
        if outcome == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if outcome == 200:
            return httpx.Response(200, json=load("cams_air_quality.json"))
        return httpx.Response(outcome)

    fc = make_client(handler).cams_forecast(*BKK)
    assert len(calls) == 4 and len(fc.rows) == 4


def test_gives_up_after_max_attempts() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    with pytest.raises(OpenMeteoError, match="HTTP 500"):
        make_client(handler, max_attempts=3).weather_forecast(*BKK)
    assert len(calls) == 3


def test_non_json_body_raises_schema_error() -> None:
    client = make_client(lambda request: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(OpenMeteoSchemaError):
        client.weather_forecast(*BKK)


def test_archive_hook_receives_raw_body_and_cannot_break_ingest() -> None:
    seen: list[RawResponse] = []
    client = make_client(json_handler("cams_air_quality.json"), archive=seen.append)
    client.cams_forecast(*BKK)
    assert seen[0].source == "cams_forecast" and seen[0].retrieved_at == NOW
    assert json.loads(seen[0].body)["_synthetic"] is True

    def boom(_: RawResponse) -> None:
        raise RuntimeError("disk full")

    assert make_client(json_handler("cams_air_quality.json"), archive=boom).cams_forecast(*BKK)


@pytest.mark.parametrize(
    ("lat", "lon", "variables"),
    [(91, 0, ["a"]), (0, 181, ["a"]), (0, 0, []), (0, 0, ["a&b=1"]), (0, 0, ["A b"])],
)
def test_invalid_inputs_rejected_before_any_request(lat, lon, variables) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ValueError):
        make_client(handler).weather_forecast(lat, lon, variables)
