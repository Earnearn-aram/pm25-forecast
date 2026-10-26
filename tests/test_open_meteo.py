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
    StationLocation,
    parse_hourly,
    to_prediction_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "open_meteo"
# Fixture hours start at 07:00 UTC; 07:30 keeps all of them inside the forecast window.
NOW = datetime(2026, 10, 26, 7, 30, tzinfo=UTC)
BKK = (13.75, 100.5)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_client(handler, **kwargs) -> OpenMeteoClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("sleep", lambda _: None)
    return OpenMeteoClient(http, retry_wait=wait_none(), **kwargs)


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


def test_cams_forecast_uses_global_domain_and_prediction_rows() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("cams_air_quality.json", calls))
    fc = client.cams_forecast(*BKK)
    assert calls[0].url.params["domains"] == "cams_global"
    rows = to_prediction_rows(fc, "stn-test", "run-1")
    assert [r.pm25_pred for r in rows] == [31.2, 33.0, 35.5]  # the null hour is skipped
    assert {(r.station_id, r.issue_time, r.model_version, r.run_id) for r in rows} == {
        ("stn-test", NOW, "cams_global", "run-1")
    }
    assert all(r.valid_time.tzinfo is not None for r in rows)
    with pytest.raises(Exception, match="frozen"):
        rows[0].pm25_pred = 1.0  # type: ignore[misc]


def test_rows_before_floor_hour_of_issue_time_are_dropped() -> None:
    later = NOW + timedelta(hours=1)  # 08:30 -> floor 08:00; the 07:00 row is analysis data
    fc = make_client(json_handler("cams_air_quality.json"), now=lambda: later).cams_forecast(*BKK)
    rows = to_prediction_rows(fc, "s", "r")
    assert [r.pm25_pred for r in rows] == [33.0, 35.5]
    assert min(r.valid_time for r in rows) == datetime(2026, 10, 26, 8, tzinfo=UTC)


def test_issue_time_is_taken_after_the_response_arrives() -> None:
    clock = iter([NOW, NOW + timedelta(seconds=5)])
    seen_by_handler: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_by_handler.append(True)
        return httpx.Response(200, json=load("cams_air_quality.json"))

    fc = make_client(handler, now=lambda: next(clock)).cams_forecast(*BKK)
    assert seen_by_handler and fc.issue_time == NOW  # first (only) clock read, post-response


def test_cams_baseline_rows_maps_stations_to_coordinates() -> None:
    calls: list[httpx.Request] = []
    stations = [
        StationLocation(station_id="a", latitude=13.7, longitude=100.5),
        StationLocation(station_id="b", latitude=13.9, longitude=100.6),
    ]
    rows = make_client(json_handler("cams_air_quality.json", calls)).cams_baseline_rows(
        stations, "run-9"
    )
    assert [c.url.params["latitude"] for c in calls] == ["13.7000", "13.9000"]
    assert {r.station_id for r in rows} == {"a", "b"} and len(rows) == 6


def test_snapshots_from_two_runs_keep_distinct_vintages() -> None:
    first = make_client(json_handler("cams_air_quality.json")).cams_forecast(*BKK)
    second_client = make_client(
        json_handler("cams_air_quality.json"), now=lambda: NOW + timedelta(minutes=20)
    )
    later = second_client.cams_forecast(*BKK)
    second_client.close()
    keys = {(r.issue_time, r.valid_time) for r in to_prediction_rows(first, "x", "r1")}
    keys |= {(r.issue_time, r.valid_time) for r in to_prediction_rows(later, "x", "r2")}
    assert len(keys) == 6  # same valid times, two issue times: nothing overwritten


def test_historical_weather_has_no_issue_time() -> None:
    calls: list[httpx.Request] = []
    client = make_client(json_handler("historical_weather.json", calls))
    fc = client.historical_weather(*BKK, date(2026, 1, 1), date(2026, 1, 2), ["temperature_2m"])
    assert fc.issue_time is None
    assert calls[0].url.params["start_date"] == "2026-01-01"
    with pytest.raises(ValueError, match="issue_time"):
        to_prediction_rows(fc, "x", "r", variable="temperature_2m")


@pytest.mark.parametrize("method", ["historical_forecast", "previous_runs"])
def test_historical_forecast_apis_have_no_issue_time_and_no_rows(method: str) -> None:
    client = make_client(json_handler("historical_weather.json"))
    fc = getattr(client, method)(*BKK, date(2026, 1, 1), date(2026, 1, 2), ["temperature_2m"])
    assert fc.issue_time is None
    with pytest.raises(ValueError, match="issue_time"):
        to_prediction_rows(fc, "x", "r", variable="temperature_2m")


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


def test_iso_naive_times_are_utc_only_when_offset_is_zero() -> None:
    hourly = {"time": ["2026-10-26T05:00"], "a": [1]}
    rows, _ = parse_hourly({"utc_offset_seconds": 0, "hourly": hourly}, ["a"])
    assert rows[0].valid_time == datetime(2026, 10, 26, 5, tzinfo=UTC)
    for payload in ({"utc_offset_seconds": 25200, "hourly": hourly}, {"hourly": hourly}):
        assert parse_hourly(payload, ["a"])[0] == []  # local/unknown zone: dropped, not guessed


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


def test_retry_after_is_honoured_and_capped() -> None:
    outcomes = iter([("429", "7"), ("429", "9999"), ("200", "")])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        code, retry_after = next(outcomes)
        if code == "200":
            return httpx.Response(200, json=load("cams_air_quality.json"))
        return httpx.Response(429, headers={"Retry-After": retry_after})

    make_client(handler, sleep=sleeps.append).cams_forecast(*BKK)
    assert sleeps == [7.0, 60.0]
