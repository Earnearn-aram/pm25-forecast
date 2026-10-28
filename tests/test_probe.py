"""Offline tests for the KAN-9 probe using synthetic, recorded-style fixtures."""

import json
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import Pki
from pm25_forecast.ingest import probe as p

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_live_tls_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default TLS diagnosis touches the network; keep every test offline."""
    monkeypatch.setattr(p.tls, "diagnose_air4thai", lambda client, requests, now=None: {})


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


def _tls_failing_handler(request: httpx.Request) -> httpx.Response:
    if "air4thai" in str(request.url):
        raise httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate",
            request=request,
        )
    return _handler(request)


def test_tls_failure_is_flagged_and_retried_with_bundle(pki: Pki, tmp_path: Path) -> None:
    from pm25_forecast.ingest import tls

    bundle = tls.der_to_pem(pki.inter_der)
    seen: list[Any] = []

    def diagnose(client: httpx.Client, requests: list, now: datetime) -> dict[str, Any]:
        assert [name for name, _, _ in requests] == ["station_list", "history"]
        return {"redirects": [], "hosts": [], "bundle_pem": bundle}

    def verified(verify: Any) -> httpx.Client:
        seen.append(verify)
        return httpx.Client(transport=httpx.MockTransport(_handler))

    report = p.run(
        tmp_path / "r",
        days=62,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_tls_failing_handler)),
        diagnose=diagnose,
        verified_client_factory=verified,
    )
    probes = {x["name"]: x for x in report["probes"]}
    assert not probes["air4thai_station_list"]["ok"]
    assert probes["air4thai_station_list"]["detail"]["tls_verify_failed"] is True
    assert not probes["air4thai_history_host"]["ok"]
    assert probes["air4thai_station_list [ca-bundle]"]["ok"]
    assert probes["air4thai_history_host [ca-bundle]"]["ok"]
    assert probes["air4thai_history_host [ca-bundle]"]["detail"]["parsed_rows"] == 4
    assert report["target_stations"][0]["station_id"] == "02t"  # data came from the verified retry
    assert len(seen) == 1 and isinstance(seen[0], ssl.SSLContext)
    assert seen[0].verify_mode == ssl.CERT_REQUIRED and seen[0].check_hostname
    # the bundle is never a trust anchor: the chain must still reach a default root
    assert not seen[0].verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    assert (tmp_path / "r.ca.pem").read_text() == bundle
    md = (tmp_path / "r.md").read_text()
    assert "Air4Thai TLS diagnosis" in md and bundle.strip() in md
    assert "station list ok=True, history host ok=True" in md


def test_no_retry_when_air4thai_works_or_no_bundle(tmp_path: Path) -> None:
    calls: list[Any] = []

    def verified(verify: Any) -> httpx.Client:
        calls.append(verify)
        return httpx.Client(transport=httpx.MockTransport(_handler))

    def diagnose(client: httpx.Client, requests: list, now: datetime) -> dict[str, Any]:
        return {"redirects": [{"name": "station_list", "status": 301, "location": "https://x/"}]}

    ok_client = lambda: httpx.Client(transport=httpx.MockTransport(_handler))  # noqa: E731
    bad_client = lambda: httpx.Client(transport=httpx.MockTransport(_tls_failing_handler))  # noqa: E731
    for factory, out in ((ok_client, "a"), (bad_client, "b")):
        report = p.run(
            tmp_path / out,
            days=5,
            pause=0,
            now=NOW,
            client_factory=factory,
            diagnose=diagnose,
            verified_client_factory=verified,
        )
        assert report["air4thai_tls"]["redirects"][0]["status"] == 301
    assert calls == []
    assert not (tmp_path / "a.ca.pem").exists() and not (tmp_path / "b.ca.pem").exists()
    assert "301" in (tmp_path / "b.md").read_text()


def test_render_tls_markdown_handles_empty_and_failed_hosts() -> None:
    assert p.render_tls_markdown({}) == []
    lines = p.render_tls_markdown(
        {
            "redirects": [{"name": "h", "status": None, "location": None, "error": "boom"}],
            "hosts": [
                {"host": "h.test", "port": 443, "error": "OSError: down", "verify_error": None}
            ],
        }
    )
    text = "\n".join(lines)
    assert "Diagnostic handshake failed: OSError: down" in text and "boom" in text


def test_history_params_use_bangkok_dates() -> None:
    params = p.history_params(NOW)
    assert params["sdate"] == "2025-12-31" and params["edate"] == "2026-01-01"
    assert params["stationID"] == "02t"


def test_render_tls_markdown_full_host_section(pki: Pki) -> None:
    from pm25_forecast.ingest import tls

    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=pki.inter_der))
    )
    host = tls.diagnose_host(
        client,
        "h.test",
        443,
        NOW,
        peek=lambda h, port: (pki.leaf_der, {"tls_version": "TLSv1.3", "cipher": "C"}),
        verify_error=lambda h, port: {"verify_code": 20, "message": "unable to get local issuer"},
        chain=lambda h, port: [tls.der_to_pem(pki.leaf_der)],
    )
    md = "\n".join(
        p.render_tls_markdown(
            {"redirects": [], "hosts": [host], "bundle_pem": tls.bundle_from_report([host])}
        )
    )
    assert "unable to get local issuer" in md
    assert "http://ca.test/inter.der" in md
    assert "matches leaf issuer: True" in md
    assert "BEGIN CERTIFICATE" in md and "certs/air4thai-ca.pem" in md


def test_diagnosis_crash_is_recorded_not_raised(tmp_path: Path) -> None:
    def explode(client: httpx.Client, requests: list, now: datetime) -> dict[str, Any]:
        raise RuntimeError("boom")

    report = p.run(
        tmp_path / "r",
        days=5,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_handler)),
        diagnose=explode,
    )
    assert report["air4thai_tls"] == {"error": "RuntimeError"}
    assert "Diagnosis failed: RuntimeError" in (tmp_path / "r.md").read_text()


def test_ca_pem_file_and_report_carry_unverified_header(pki: Pki, tmp_path: Path) -> None:
    from pm25_forecast.ingest import tls

    host = {
        "issuer_downloads": [
            {
                "url": "http://ca.test/inter.der",
                "pem": tls.der_to_pem(pki.inter_der),
                "sha256": "cd" * 32,
                "fetched_at": NOW.isoformat(),
            }
        ]
    }
    bundle = tls.bundle_from_report([host])
    report = p.run(
        tmp_path / "r",
        days=5,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_tls_failing_handler)),
        diagnose=lambda c, r, n: {"hosts": [host], "bundle_pem": bundle},
        verified_client_factory=lambda v: httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    assert report["air4thai_tls"]["retry_with_bundle"]
    for text in ((tmp_path / "r.ca.pem").read_text(), (tmp_path / "r.md").read_text()):
        assert "# UNVERIFIED: fetched over plain http from http://ca.test/inter.der" in text
        assert "SHA-256 (DER): " + "cd" * 32 in text


def test_presented_count_rendering() -> None:
    assert "openssl unavailable" in p._presented_text(None)
    assert "connection failed" in p._presented_text(0)
    assert p._presented_text(2) == "2"


# ---------------------------------------------------------------- KAN-9 history retention

RETAINED = "air4thai_history_retained.json"  # the endpoint's whole retained archive for 02t
EARLIEST = datetime(2025, 12, 19, 17, tzinfo=UTC)  # 2025-12-20 00:00 Asia/Bangkok


def _retention_handler(request: httpx.Request) -> httpx.Response:
    """Serve history like Air4Thai: rows inside the retained window, else 200 with no rows."""
    if "getHistoryData" not in str(request.url):
        return _handler(request)
    q = request.url.params
    archive = load(RETAINED)
    rows = [
        r for r in archive["stations"][0]["data"] if q["sdate"] <= r["DATETIC"][:10] <= q["edate"]
    ]
    if not rows:
        return httpx.Response(200, json=load("air4thai_history_empty.json"))
    archive["stations"][0]["data"] = rows
    return httpx.Response(200, json=archive)


def test_retention_detected_from_empty_old_chunks(tmp_path: Path) -> None:
    report = p.run(
        tmp_path / "r",
        days=62,
        chunk_days=10,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_retention_handler)),
    )
    h = report["air4thai_history"]
    assert h["stations"]["02t"]["chunks_empty"] == 4  # Nov 1 .. Dec 11 local: 200, no rows
    assert h["stations"]["02t"]["chunks_with_rows"] == 3
    assert h["stations"]["02t"]["chunks_failed"] == 0
    assert h["earliest"] == EARLIEST
    assert h["latest"] == datetime(2026, 1, 2, 11, tzinfo=UTC)
    assert h["retention_days"] == 13.8  # EARLIEST -> NOW (requested end)
    assert h["requested_days"] == 62.0
    assert h["truncated"] and h["rolling_window"]
    assert h["missing_pct_in_window"] == {"02t": 0.3}  # 1 null hour in 331
    assert h["stations"]["02t"]["missing_pct_requested"] == 77.82  # 330 of 1488 hours
    assert h["complete_in_window"] == ["02t"]
    assert report["target_stations"][0]["missing_pct_in_window"] == 0.3
    hist = next(x for x in report["probes"] if x["name"] == "air4thai_history_endpoint")
    assert hist["detail"]["requests_ok_empty"] == 4
    d = report["decision"]
    assert d["decision"] == "NO-GO"  # thresholds and logic unchanged
    assert d["reason"].startswith(
        "Air4Thai history is a rolling ~14-day window (earliest 2025-12-19T17:00:00+00:00; "
        "4 chunk requests returned HTTP 200 with no rows); 2-year (>= 700-day) backfill "
        "impossible from Air4Thai; stations complete within window (<= 15.0% missing): 02t; "
    )
    assert "neither source meets the bar (0 good Air4Thai stations" in d["reason"]
    md = (tmp_path / "r.md").read_text()
    assert "## Air4Thai history retention" in md
    assert "| 02t | 3 | 4 | 0 | 2025-12-19 17:00:00+00:00 | 77.82 | 0.3 |" in md
    assert "Rolling retention window detected: True" in md


def _hist_cov(
    first: datetime, hours: int, empty: int, start: datetime, end: datetime
) -> p.Coverage:
    series: dict[datetime, float | None] = {first + timedelta(hours=h): 1.0 for h in range(hours)}
    cov = p.compute_coverage(series, start, end)
    return cov.model_copy(update={"chunks_empty": empty, "chunks_with_rows": 1})


def test_summarize_history_pure() -> None:
    end = datetime(2026, 10, 26, tzinfo=UTC)
    start = end - timedelta(days=730)
    first = datetime(2026, 7, 31, 17, tzinfo=UTC)
    full = int((end - first).total_seconds() // 3600)
    cov = {
        "59t": _hist_cov(first, full, 21, start, end),
        "bad": _hist_cov(first, full // 2, 21, start, end),
    }
    r = p.summarize_history(cov)
    assert r.earliest == first and r.retention_days == 86.3
    assert r.missing_pct_in_window == {"59t": 0.0, "bad": 50.02}  # 1035 of 2071 hours
    assert r.complete_in_window == ["59t"]
    assert r.rolling_window and r.chunks_empty == 42
    assert cov["59t"].missing_pct > 85  # the old "missing 88%" measure, still reported
    text = p.retention_text(r)
    assert "rolling ~86-day window (earliest 2026-07-31T17:00:00+00:00" in text
    assert text.endswith("stations complete within window (<= 15.0% missing): 59t")
    # no retention diagnosis when history covers the requested window or nothing came back
    whole = {"a": _hist_cov(start, 730 * 24, 0, start, end)}
    assert not p.summarize_history(whole).truncated
    assert p.retention_text(p.summarize_history(whole)) == ""
    assert p.retention_text(p.summarize_history({})) == ""
    assert p.decide({"a": _cov(10, 50)}, [], NOW)["reason"].startswith("neither source")
    # truncated but old chunks failed instead of returning empty: not called "rolling"
    failed = {"a": cov["59t"].model_copy(update={"chunks_empty": 0, "chunks_failed": 3})}
    assert "only a ~86-day window" in p.retention_text(p.summarize_history(failed))
    assert p.render_history_markdown({}) == []


# ---------------------------------------------------------------- KAN-46 OpenAQ target coverage
# All payloads below are SYNTHETIC (see the "_note" in each fixture); no network.

OAQ_FIRST = datetime(2025, 6, 1, 0, tzinfo=UTC)
OAQ_LAST = datetime(2026, 1, 2, 10, tzinfo=UTC)


def _target_stations() -> dict[str, p.Station]:
    parsed = p.parse_air4thai_stations(load("air4thai_stations_targets_synthetic.json"))
    return {s.station_id: s for s in parsed}


def _oaq_handler(request: httpx.Request) -> httpx.Response:
    url = request.url
    path = url.path
    if "getNewAQI_JSON" in str(url):
        return httpx.Response(200, json=load("air4thai_stations_targets_synthetic.json"))
    if "openaq" not in str(url):
        return httpx.Response(404)
    assert request.headers["X-API-Key"] == "dummy-value"
    if path.endswith("/v3/locations"):
        assert url.params["parameters_id"] == "2"  # PM2.5 only
        return httpx.Response(200, json=load("openaq_target_locations_synthetic.json"))
    parts = path.strip("/").split("/")  # v3/locations/<id>/sensors or v3/sensors/<id>/hours
    if parts[1] == "locations" and parts[-1] == "sensors":
        return httpx.Response(200, json=load("openaq_target_sensors_synthetic.json")[parts[2]])
    if parts[1] == "sensors" and parts[-1] == "hours":
        return httpx.Response(200, json=load("openaq_target_hours_synthetic.json")[parts[2]])
    return httpx.Response(404)


def test_target_constant_includes_bkp101t() -> None:
    assert p.TARGET_STATION_IDS == ("59t", "61t", "bkp100t", "bkp101t", "02t")


def test_openaq_parsers_pm25_only_and_tolerant() -> None:
    sensors = load("openaq_target_sensors_synthetic.json")
    assert p.parse_openaq_sensors(sensors["201"]) == [9001]  # pm10, bad id, garbage dropped
    assert p.parse_openaq_sensors(None) == []
    hours = p.parse_openaq_hours(load("openaq_target_hours_synthetic.json")["9001"])
    assert hours[OAQ_FIRST] == 10.0
    assert hours[OAQ_FIRST + timedelta(hours=4)] is None
    assert len(hours) == 5 and all(t.tzinfo for t in hours)  # malformed row dropped
    assert p.parse_openaq_hours({"results": "x"}) == {}


def test_rank_candidates_nearest_first_with_runner_up() -> None:
    stations = _target_stations()
    locs = p.parse_openaq_locations(load("openaq_target_locations_synthetic.json"))
    ranked = p.rank_openaq_candidates(stations["59t"], locs)
    assert [c.location_id for c in ranked] == [202, 203]
    assert ranked[0].distance_m < 50 < ranked[1].distance_m < p.MATCH_RADIUS_M
    assert p.rank_openaq_candidates(stations["61t"], locs) == []
    assert p.rank_openaq_candidates(p.Station(station_id="x"), locs) == []


def test_summarize_openaq_series_window_and_span() -> None:
    hours = p.parse_openaq_hours(load("openaq_target_hours_synthetic.json")["9001"])
    end = NOW.replace(minute=0)
    cov = p.summarize_openaq_series("02t", hours, end - timedelta(days=730), end)
    span_hours = int((OAQ_LAST - OAQ_FIRST).total_seconds() // 3600) + 1
    assert (cov.status, cov.hours_with_data, cov.window_hours) == ("ok", 4, 17520)
    assert (cov.first, cov.last) == (OAQ_FIRST, OAQ_LAST)
    assert cov.missing_pct_window == round(100 * (1 - 4 / 17520), 2)
    assert cov.missing_pct_span == round(100 * (1 - 4 / span_hours), 2)
    assert cov.span_days == round((OAQ_LAST - OAQ_FIRST).total_seconds() / 86400, 1)
    empty = p.summarize_openaq_series("x", {}, end - timedelta(days=730), end)
    assert empty.status == "no_data" and empty.missing_pct_window == 100.0
    assert empty.missing_pct_span is None


def _run_oaq(tmp_path: Path, handler: Any) -> dict[str, Any]:
    return p.run(
        tmp_path / "r",
        days=5,
        pause=0,
        now=NOW,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_run_reports_openaq_coverage_for_all_five_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy-value")
    report = _run_oaq(tmp_path, _oaq_handler)
    section = report["openaq_target_coverage"]
    assert section["targets"] == ["59t", "61t", "bkp100t", "bkp101t", "02t"]
    rows = {r["station_id"]: r for r in section["stations"]}
    assert list(rows) == section["targets"]
    assert rows["02t"]["status"] == "ok"
    assert rows["02t"]["matched"]["location_id"] == 201
    assert rows["02t"]["sensor_ids"] == [9001]  # PM2.5 sensor only, not the PM10 one
    assert rows["02t"]["first"] == OAQ_FIRST and rows["02t"]["last"] == OAQ_LAST
    assert rows["59t"]["matched"]["location_id"] == 202
    assert rows["59t"]["runner_up"]["location_id"] == 203
    assert rows["59t"]["status"] == "no_data"  # the sensor returned an empty page
    assert rows["61t"]["status"] == "no_match" and rows["61t"]["matched"] is None
    assert rows["bkp100t"]["status"] == "no_pm25_sensor"
    assert rows["bkp101t"]["matched"]["location_id"] == 204  # no current PM2.5 in Air4Thai list
    assert rows["bkp101t"]["status"] == "no_data"  # only null values returned
    md = (tmp_path / "r.md").read_text()
    assert "## OpenAQ coverage for target stations" in md
    assert "no match within 1000 m" in md
    assert all(f"| {sid} |" in md for sid in section["targets"])
    # the key never reaches the report files; the GO/NO-GO decision is still produced
    assert all("dummy-value" not in (tmp_path / n).read_text() for n in ("r.json", "r.md"))
    assert report["decision"]["decision"] in {"GO-air4thai", "GO-openaq", "NO-GO"}


def test_openaq_coverage_skipped_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    report = _run_oaq(tmp_path, _oaq_handler)
    rows = report["openaq_target_coverage"]["stations"]
    assert [r["status"] for r in rows] == ["skipped"] * 5
    assert "OPENAQ_API_KEY not set; skipped" in (tmp_path / "r.md").read_text()


def test_openaq_429_is_retried_and_pagination_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p, "OPENAQ_PAGE_LIMIT", 2)
    monkeypatch.setattr(p.time, "sleep", lambda s: None)
    seen: dict[str, Any] = {"limited": 0, "pages": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/locations") and not seen["limited"]:
            seen["limited"] = 1
            return httpx.Response(429, headers={"Retry-After": "0"})
        if request.url.path.endswith("/hours"):
            page = int(request.url.params["page"])
            seen["pages"].append(page)
            rows = load("openaq_target_hours_synthetic.json")["9001"]["results"]
            return httpx.Response(200, json={"results": rows[(page - 1) * 2 : page * 2]})
        return _oaq_handler(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = p.probe_openaq_coverage(
        client, "dummy-value", _target_stations(), NOW, 0, targets=["02t"]
    )
    assert seen["limited"] == 1
    assert seen["pages"] == [1, 2, 3, 4]  # 6 rows at 2 per page; the empty 4th page stops it
    assert rows[0].status == "ok" and rows[0].hours_with_data == 4


def test_openaq_failures_are_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p.time, "sleep", lambda s: None)

    def always_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"})

    client = httpx.Client(transport=httpx.MockTransport(always_429))
    rows = p.probe_openaq_coverage(
        client, "dummy-value", _target_stations(), NOW, 0, targets=["02t", "zz"]
    )
    assert rows[0].status == "error" and "HTTP 429" in rows[0].note
    assert rows[1].status == "station_unknown"
    assert "dummy-value" not in rows[0].model_dump_json()
