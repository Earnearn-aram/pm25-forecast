"""Air4Thai client tests on a synthetic fixture. No live calls: transport is mocked."""

import gzip
import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from pm25_forecast.ingest import air4thai
from pm25_forecast.ingest.air4thai import (
    Air4ThaiClient,
    Air4ThaiError,
    Observation,
    gzip_file_archive,
    parse_local_timestamp,
    parse_response,
)
from pm25_forecast.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "air4thai_getNewAQI_synthetic.json"


@pytest.fixture
def body() -> bytes:
    return FIXTURE.read_bytes()


def make_client(handler, **kwargs) -> Air4ThaiClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return Air4ThaiClient(client=http, wait_max=0.0, **kwargs)


def test_local_time_converted_to_utc() -> None:
    ts = parse_local_timestamp("2026-10-26", "08:00")
    assert ts == datetime(2026, 10, 26, 1, 0, tzinfo=UTC)
    assert parse_local_timestamp("2026-10-26", "24:00") == datetime(2026, 10, 26, 17, 0, tzinfo=UTC)


def test_parse_tolerates_gaps(body: bytes, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        obs = {o.station_id: o for o in parse_response(json.loads(body))}
    assert obs["TST01"].pm25 == 17.5
    assert obs["TST01"].ts == datetime(2026, 10, 26, 1, 0, tzinfo=UTC)
    assert obs["TST01"].source == "air4thai"
    assert obs["TST02"].pm25 is None  # -1 sentinel
    assert obs["TST04"].pm25 is None  # empty string
    assert "TST05" not in obs and "TST06" not in obs
    assert sum("invalid station entry" in r.message for r in caplog.records) == 2


def test_missing_pm25_block_kept_as_gap(body: bytes) -> None:
    obs = {o.station_id: o for o in parse_response(json.loads(body))}
    assert obs["TST03"].pm25 is None


@pytest.mark.parametrize("payload", [[], {}, {"stations": "x"}, None])
def test_unusable_top_level_raises(payload: object) -> None:
    with pytest.raises(Air4ThaiError):
        parse_response(payload)


def test_observation_rejects_naive_or_non_utc() -> None:
    with pytest.raises(ValueError):
        Observation(station_id="a", ts=datetime(2026, 1, 1, tzinfo=None), pm25=1.0)  # noqa: DTZ001


def test_fetch_filters_stations_and_archives(body: bytes, tmp_path: Path) -> None:
    client = make_client(
        lambda r: httpx.Response(200, content=body), archive=gzip_file_archive(tmp_path)
    )
    obs = client.fetch(["TST01", "NOPE"])
    assert [o.station_id for o in obs] == ["TST01"]
    files = list(tmp_path.glob("air4thai-*.json.gz"))
    assert len(files) == 1
    assert gzip.decompress(files[0].read_bytes()) == body


def test_retries_then_succeeds(body: bytes) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("t", request=request)
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=body)

    assert len(make_client(handler).fetch()) == 4
    assert calls["n"] == 3


def test_no_retry_on_client_error_and_gives_up_after_max() -> None:
    calls = {"n": 0}

    def not_found(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(not_found).fetch()
    assert calls["n"] == 1

    calls["n"] = 0

    def boom(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(boom, max_attempts=3).fetch()
    assert calls["n"] == 3


def test_invalid_json_raises() -> None:
    with pytest.raises(Air4ThaiError):
        make_client(lambda r: httpx.Response(200, content=b"<html>")).fetch()


def test_archive_failure_does_not_block_ingest(body: bytes) -> None:
    def bad_archive(raw: bytes, at: datetime) -> None:
        raise OSError("disk full")

    client = make_client(lambda r: httpx.Response(200, content=body), archive=bad_archive)
    assert len(client.fetch()) == 4


def test_ingest_calls_sink(body: bytes) -> None:
    received: list[Observation] = []
    client = make_client(lambda r: httpx.Response(200, content=body))
    assert client.ingest(received.extend, ["TST01"]) == 1
    assert received[0].station_id == "TST01"


def test_default_timeout_is_set() -> None:
    assert air4thai.DEFAULT_TIMEOUT.read is not None


def test_station_ids_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIR4THAI_STATION_IDS", "a1, b2,,")
    assert Settings(_env_file=None).station_id_list == ["a1", "b2"]


def test_all_stations_unparseable_raises() -> None:
    with pytest.raises(Air4ThaiError):
        parse_response({"stations": [{"nameEN": "no id"}, {"stationID": "x", "AQILast": {}}]})


def test_empty_stations_list_is_not_an_error() -> None:
    assert parse_response({"stations": []}) == []


def test_all_pm25_none_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    payload = {"stations": [{"stationID": "a", "AQILast": {"date": "2026-10-26", "time": "08:00"}}]}
    with caplog.at_level(logging.ERROR):
        obs = parse_response(payload)
    assert obs[0].pm25 is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.mark.parametrize("bad", ["99:00", "12:60", "25:00", "-1:00", "12", "1:2:3:4", "ab:cd"])
def test_bad_times_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_local_timestamp("2026-10-26", bad)


def test_time_with_seconds_accepted() -> None:
    assert parse_local_timestamp("2026-10-26", "08:30:15") == datetime(
        2026, 10, 26, 1, 30, 15, tzinfo=UTC
    )


def test_aware_non_utc_ts_rejected() -> None:
    bkk = timezone(timedelta(hours=7))
    with pytest.raises(ValueError):
        Observation(station_id="a", ts=datetime(2026, 1, 1, tzinfo=bkk), pm25=1.0)


def test_archive_unexpected_exception_does_not_block(body: bytes) -> None:
    def bad(raw: bytes, at: datetime) -> None:
        raise RuntimeError("boom")

    client = make_client(lambda r: httpx.Response(200, content=body), archive=bad)
    assert len(client.fetch()) == 4


def test_close_only_closes_owned_client() -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    Air4ThaiClient(client=http).close()
    assert not http.is_closed
    owned = Air4ThaiClient()
    owned.close()
    assert owned._client.is_closed
    assert owned._client.follow_redirects


def test_default_archive_dir_is_gitignored() -> None:
    assert Path("data/raw/air4thai") == air4thai.DEFAULT_ARCHIVE_DIR
    assert "*.json.gz" in (Path(__file__).parent.parent / ".gitignore").read_text()


def test_certificate_verify_failures_are_not_retried() -> None:
    tls_error = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer")
    assert air4thai._is_retryable(tls_error) is False
    assert air4thai._is_retryable(httpx.ConnectError("connection reset")) is True
