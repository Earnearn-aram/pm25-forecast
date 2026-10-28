"""Synthetic HTTP and SQLite only: no OpenAQ credentials or live calls."""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import httpx
import pytest

from pm25_forecast import openaq_backfill as b
from pm25_forecast.collector import ConfigError
from pm25_forecast.ingest.probe import TARGET_STATION_IDS
from pm25_forecast.settings import Settings
from pm25_forecast.storage import sql
from test_collector import DB_URL, FakeDb

END = datetime(2026, 10, 28, tzinfo=UTC)
START = END - timedelta(days=1)


def settings():
    return Settings(
        _env_file=None,
        database_url=DB_URL,
        station_locations=";".join(
            f"{sid}={13 + i / 10},100" for i, sid in enumerate(TARGET_STATION_IDS)
        ),
    )


def hour(value=10, stamp=START):
    return {"value": value, "period": {"datetimeFrom": {"utc": stamp.isoformat()}}}


def sensor(sid):
    return {"id": sid, "parameter": {"id": 2}}


def handler(request):
    path = request.url.path
    if path.endswith("/locations"):
        lat = float(request.url.params["coordinates"].split(",")[0])
        i = round((lat - 13) * 10)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 100 + i,
                        "coordinates": {"latitude": lat, "longitude": 100},
                        "sensors": [sensor(900 + i), {"id": 99, "parameter": {"id": 1}}],
                    }
                ]
            },
        )
    assert "/sensors/" in path and path.endswith("/hours")
    return httpx.Response(200, json={"results": [hour()]})


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "synthetic-test-value")


def run(fake, responder=handler, **kw):
    with httpx.Client(transport=httpx.MockTransport(responder)) as http:
        return b.run(
            settings(),
            connect=fake,
            http=http,
            sleep=lambda _: None,
            start=kw.pop("start", START),
            end=kw.pop("end", END),
            **kw,
        )


def test_five_station_embedded_mapping_and_idempotent_rerun():
    fake = FakeDb()
    reports = run(fake)
    assert [r.station_id for r in reports] == list(TARGET_STATION_IDS)
    assert [r.location_id for r in reports] == list(range(100, 105))
    assert [r.sensor_ids for r in reports] == [[900 + i] for i in range(5)]
    assert sum(r.rows_new for r in reports) == 5
    assert sum(r.rows_new for r in run(fake)) == 0
    assert len(fake.rows("observations")) == 5
    assert all(row[3] == "openaq" for row in fake.rows("observations"))
    assert fake.commits == 10


def test_multiple_sensors_deterministic_priority_and_validation():
    def responder(request):
        if request.url.path.endswith("/locations"):
            payload = handler(request).json()
            payload["results"][0]["sensors"] = [sensor(902), sensor(901), sensor(902)]
            return httpx.Response(200, json=payload)
        value = 22 if "/902/" in request.url.path else 11
        return httpx.Response(
            200,
            json={
                "results": [
                    hour(value),
                    hour(None, START + timedelta(hours=1)),
                    hour(-1, START + timedelta(hours=2)),
                    hour(1001, START + timedelta(hours=3)),
                    hour("Infinity", START + timedelta(hours=4)),
                    hour(4, START.replace(tzinfo=None)),
                    hour(4, START + timedelta(minutes=30)),
                    hour(4, END),
                    hour(4, START - timedelta(hours=1)),
                ]
            },
        )

    fake = FakeDb()
    report = run(fake, responder, station_ids=["59t"])[0]
    assert report.sensor_ids == [901, 902]
    assert fake.rows("observations")[0][2:4] == (22, "openaq")
    assert len(fake.rows("observations")) == 1


def test_partial_sensor_failure_keeps_previous_chunk_and_other_stations():
    fake = FakeDb()
    run(fake)
    before = fake.rows("observations")

    def responder(request):
        if request.url.path.endswith("/locations"):
            payload = handler(request).json()
            if payload["results"][0]["id"] == 100:
                payload["results"][0]["sensors"] += [sensor(999)]
            return httpx.Response(200, json=payload)
        if "/999/" in request.url.path:
            return httpx.Response(503)
        return httpx.Response(200, json={"results": [hour(99)]})

    reports = run(fake, responder)
    assert reports[0].failures and reports[0].chunks == 0
    assert next(r for r in fake.rows("observations") if r[0] == "59t") == next(
        r for r in before if r[0] == "59t"
    )
    assert all(r.rows_upserted == 1 for r in reports[1:])
    assert run(fake)[0].failures == []  # repair by replay


def test_pagination_and_compatibility_fallback(monkeypatch):
    monkeypatch.setattr(b, "OPENAQ_PAGE_LIMIT", 2)
    calls = []

    def responder(request):
        calls.append((request.url.path, request.url.params.get("page")))
        if request.url.path.endswith("/locations"):
            payload = handler(request).json()
            payload["results"][0].pop("sensors")
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/sensors"):
            return httpx.Response(200, json={"results": [sensor(900)]})
        page = int(request.url.params["page"])
        return httpx.Response(
            200,
            json={
                "results": [
                    hour(10, START + timedelta(hours=i)) for i in ([0, 1] if page == 1 else [2])
                ]
            },
        )

    fake = FakeDb()
    assert run(fake, responder, station_ids=["59t"])[0].rows_new == 3
    assert ("/v3/sensors/900/hours", "2") in calls


def test_chunk_transactions_survive_interruption():
    fake = FakeDb()
    calls = 0

    def responder(request):
        nonlocal calls
        if request.url.path.endswith("/hours"):
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            stamp = datetime.fromisoformat(request.url.params["datetime_from"])
            return httpx.Response(200, json={"results": [hour(10, stamp)]})
        return handler(request)

    start = END - timedelta(days=730)
    with pytest.raises(KeyboardInterrupt):
        run(fake, responder, station_ids=["59t"], start=start)
    assert fake.commits == 1 and len(fake.rows("observations")) == 1
    assert fake.rows("observations")[0][1] == start.isoformat()


@pytest.mark.parametrize("order", [("air4thai", "openaq"), ("openaq", "air4thai")])
def test_overlap_precedence_both_arrival_orders(order):
    fake = FakeDb()
    row = {"station_id": "59t", "ts": START, "pm25": 10}
    for source in order:
        fake.execute(
            sql.UPSERT_OBSERVATION,
            {**row, "source": source, "pm25": 20 if source == "air4thai" else 10},
        )
    assert fake.rows("observations")[0][2:4] == (20, "air4thai")
    fake.execute(sql.UPSERT_OBSERVATION, {**row, "source": "openaq", "pm25": 99})
    assert fake.rows("observations")[0][2:4] == (20, "air4thai")


def test_null_air4thai_gap_fill_preserves_provenance():
    fake = FakeDb()
    row = {"station_id": "59t", "ts": START, "pm25": None, "source": "air4thai"}
    fake.execute(sql.UPSERT_OBSERVATION, row)
    fake.execute(sql.UPSERT_OBSERVATION, {**row, "source": "openaq", "pm25": 12})
    fake.execute(sql.UPSERT_OBSERVATION, row)
    assert fake.rows("observations")[0][2:4] == (12, "openaq")
    fake.execute(sql.UPSERT_OBSERVATION, {**row, "pm25": 13})
    assert fake.rows("observations")[0][2:4] == (13, "air4thai")


def test_missing_key_ignores_settings_key_before_io(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY")
    config = settings()
    config.openaq_api_key = config.database_url  # synthetic SecretStr; environment still required
    with pytest.raises(ConfigError, match="OPENAQ_API_KEY"):
        b.run(config, connect=lambda _: pytest.fail("must not connect"))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_bounded_retry_and_secret_safe_errors(status, caplog):
    sleeps, calls = [], []

    def responder(request):
        calls.append(request)
        return httpx.Response(
            status, headers={"Retry-After": "999999"}, text="synthetic-test-value"
        )

    with httpx.Client(transport=httpx.MockTransport(responder)) as http:
        client = b.OpenAQClient(http, "synthetic-test-value", sleeps.append)
        with pytest.raises(b.OpenAQError, match=f"HTTP {status}"):
            client.get("/locations", {})
    assert len(calls) == b.MAX_ATTEMPTS
    assert max(sleeps) <= b.MAX_WAIT
    assert "synthetic-test-value" not in caplog.text


def test_retry_recovers_and_non_retryable_status_is_immediate():
    statuses = iter([429, 503, 200, 401])
    sleeps = []
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(next(statuses), json={"results": []})
        )
    ) as http:
        client = b.OpenAQClient(http, "synthetic-test-value", sleeps.append)
        assert client.get("/locations", {}).results == []
        with pytest.raises(b.OpenAQError, match="401"):
            client.get("/locations", {})
    assert [s for s in sleeps if s > 1] == [2.0]


@pytest.mark.parametrize("payload", [None, {}, {"results": None}])
def test_malformed_payload_is_failure(payload):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ) as http,
        pytest.raises(b.OpenAQError, match="invalid response"),
    ):
        b.OpenAQClient(http, "synthetic-test-value", lambda _: None).get("/locations", {})


def test_page_limit_fails_instead_of_silent_truncation(monkeypatch):
    monkeypatch.setattr(b, "MAX_PAGES", 2)
    monkeypatch.setattr(b, "OPENAQ_PAGE_LIMIT", 1)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": [hour()]}))
        ) as http,
        pytest.raises(b.OpenAQError, match="pagination limit"),
    ):
        list(b.OpenAQClient(http, "synthetic-test-value", lambda _: None).pages("/x", {}))


def test_cli_missing_key_and_partial_failure(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY")
    assert b.main([]) == 2
    monkeypatch.setattr(
        b, "run", lambda *a, **kw: [b.Report(station_id="59t", failures=["HTTP 503"])]
    )
    assert b.main([]) == 1
    monkeypatch.setattr(b, "run", lambda *a, **kw: [b.Report(station_id="59t", rows_usable=1)])
    assert b.main([]) == 0


def test_complete_730_day_window_has_bounded_transactions_and_no_duplicate_boundary_hours():
    fake = FakeDb()
    windows = []

    def responder(request):
        if request.url.path.endswith("/hours"):
            start = datetime.fromisoformat(request.url.params["datetime_from"])
            end = datetime.fromisoformat(request.url.params["datetime_to"])
            windows.append((start, end))
            # Include the API's end boundary deliberately: the client must exclude it.
            return httpx.Response(200, json={"results": [hour(10, start), hour(99, end)]})
        return handler(request)

    start = END - timedelta(days=730)
    report = run(fake, responder, station_ids=["59t"], start=start)[0]
    assert report.chunks == 24 and report.rows_new == 24
    assert windows[0][0] == start and windows[-1][1] == END
    assert all(end - start <= timedelta(days=31) for start, end in windows)
    assert all(a[1] == c[0] for a, c in pairwise(windows))
    assert run(fake, responder, station_ids=["59t"], start=start)[0].rows_new == 0
    assert fake.commits == 48


def test_all_air4thai_overlap_is_successful_usable_run():
    fake = FakeDb()
    for sid in TARGET_STATION_IDS:
        fake.execute(
            sql.UPSERT_OBSERVATION,
            {
                "station_id": sid,
                "ts": START,
                "pm25": 40,
                "source": "air4thai",
            },
        )
    reports = run(fake)
    assert all(r.rows_upserted == 0 and r.rows_usable == 1 and not r.failures for r in reports)
    assert all(row[2:4] == (40, "air4thai") for row in fake.rows("observations"))


def test_transport_failure_is_bounded_and_sanitized():
    calls = []

    def responder(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic-test-value", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(responder)) as http,
        pytest.raises(b.OpenAQError, match="transport failure") as error,
    ):
        b.OpenAQClient(http, "synthetic-test-value", lambda _: None).get("/locations", {})
    assert len(calls) == b.MAX_ATTEMPTS
    assert "synthetic-test-value" not in str(error.value)


@pytest.mark.parametrize(
    "start,end,ids",
    [
        (END, START, ["59t"]),
        (START.replace(tzinfo=None), END, ["59t"]),
        (START, END + timedelta(minutes=1), ["59t"]),
        (START, END, ["unknown"]),
        (START, END, []),
    ],
)
def test_bad_configuration_fails_before_io(start, end, ids):
    with pytest.raises(ConfigError):
        b.run(
            settings(),
            start=start,
            end=end,
            station_ids=ids,
            connect=lambda _: pytest.fail("must not connect"),
        )


def test_database_failure_exits_nonzero_without_echoing_exception(monkeypatch, caplog):
    def broken(*a, **kw):
        raise RuntimeError("synthetic-test-value")

    monkeypatch.setattr(b, "run", broken)
    assert b.main([]) == 1
    assert "synthetic-test-value" not in caplog.text


def test_retry_after_http_date_and_bad_header():
    future = END + timedelta(days=36500)
    response = httpx.Response(
        429, headers={"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")}
    )
    assert b.retry_wait(response, 0) == b.MAX_WAIT
    assert b.retry_wait(httpx.Response(429), 2) == 4
