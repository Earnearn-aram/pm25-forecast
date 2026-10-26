"""Offline tests for the KAN-9 probe using synthetic, recorded-style fixtures."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from pm25_forecast.ingest import probe as p

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text())


def test_station_parse_and_bangkok_filter() -> None:
    stations = p.parse_air4thai_stations(load("air4thai_stations.json"))
    assert {s.station_id for s in stations} == {"02t", "03t", "50t", "bad"}
    bangkok = p.filter_bangkok(stations)
    assert [s.station_id for s in bangkok] == ["02t"]  # 03t reports -1, bad has no PM2.5
    assert p.parse_air4thai_stations("nope") == []


def test_history_parse_converts_local_to_utc_and_tolerates_gaps() -> None:
    series = p.parse_air4thai_history(load("air4thai_history.json"))
    assert series[datetime(2026, 1, 1, 0, tzinfo=UTC)] == 30.0
    assert series[datetime(2026, 1, 1, 1, tzinfo=UTC)] is None
    assert len(series) == 4  # malformed timestamp dropped
    assert p.parse_air4thai_history({"stations": None}) == {}


def test_coverage_missing_pct_and_gaps() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    series: dict[datetime, float | None] = {
        start + timedelta(hours=h): 10.0 for h in (0, 1, 2, 6, 7)
    }
    series[start + timedelta(hours=3)] = None
    cov = p.compute_coverage(series, start, start + timedelta(hours=10))
    assert cov.hours_with_data == 5
    assert cov.missing_pct == 50.0
    assert cov.longest_gap_hours == 3  # hours 3-5
    empty = p.compute_coverage({}, start, start + timedelta(hours=10))
    assert empty.missing_pct == 100.0
    assert empty.first is None


def test_open_meteo_summary() -> None:
    s = p.parse_open_meteo(load("open_meteo_air_quality.json"), "pm2_5")
    assert (s["total"], s["non_null"]) == (4, 2)
    assert s["first_non_null"] == "2026-01-01T02:00"
    assert p.parse_open_meteo({"error": True}, "pm2_5")["variable_present"] is False


def test_openaq_parse_and_station_match() -> None:
    locs = p.parse_openaq_locations(load("openaq_locations.json"))
    assert [loc.id for loc in locs] == [101, 102]
    stations = p.parse_air4thai_stations(load("air4thai_stations.json"))
    matches = p.match_openaq_to_air4thai(stations, locs)
    assert matches["02t"] == 101
    assert matches["03t"] is None


def _cov(span: float, missing: float) -> p.Coverage:
    return p.Coverage(expected_hours=1, hours_with_data=1, missing_pct=missing, span_days=span)


def test_decide_branches() -> None:
    good = {k: _cov(720, 5) for k in "abc"}
    assert p.decide(good, [], NOW)["decision"] == "GO-air4thai"
    deep = [p.OpenAQLocation(id=i, lat=1, lon=1, first=NOW - timedelta(days=900)) for i in range(3)]
    assert p.decide({"a": _cov(10, 50)}, deep, NOW)["decision"] == "GO-openaq"
    assert p.decide({"a": _cov(10, 50)}, [], NOW)["decision"] == "NO-GO"
    assert p.decide({}, [], NOW)["decision"] == "UNDETERMINED"


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "getNewAQI_JSON" in url:
        return httpx.Response(200, json=load("air4thai_stations.json"))
    if "getHistoryData" in url:
        return httpx.Response(200, json=load("air4thai_history.json"))
    if "open-meteo" in url and "historical-forecast" in url:
        return httpx.Response(404, json={"error": True})
    if "open-meteo" in url:
        return httpx.Response(200, json=load("open_meteo_air_quality.json"))
    if "openaq" in url:
        assert request.headers["X-API-Key"] == "dummy-value"
        return httpx.Response(200, json=load("openaq_locations.json"))
    return httpx.Response(500)


def test_run_end_to_end_with_mock_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy-value")
    report = p.run(
        tmp_path / "r",
        days=62,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    names = {x["name"]: x for x in report["probes"]}
    assert names["air4thai_station_list"]["ok"]
    assert names["air4thai_history_endpoint"]["ok"]
    assert not names["open_meteo:historical_forecast_api"]["ok"]
    assert names["open_meteo:air_quality_api"]["ok"]
    assert report["openaq_same_stations"] == {"matched": 1, "of": 1}
    assert report["target_stations"][0]["station_id"] == "02t"
    assert (tmp_path / "r.json").exists()
    assert "KAN-9 API probe report" in (tmp_path / "r.md").read_text()
    assert "dummy-value" not in (tmp_path / "r.json").read_text()


def test_run_records_network_failure_and_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("blocked", request=request)

    report = p.run(
        tmp_path / "r",
        days=5,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(boom)),
    )
    assert report["decision"]["decision"] == "UNDETERMINED"
    assert all(not x["ok"] for x in report["probes"])
    assert any("skipped" in (x["error"] or "") for x in report["probes"])


REAL_CLIENT = httpx.Client


def test_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        p.httpx,
        "Client",
        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(_handler)),
    )
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    assert p.main(["--out", str(tmp_path / "x"), "--days", "35", "--pause", "0"]) == 0
    assert (tmp_path / "x.md").exists()
