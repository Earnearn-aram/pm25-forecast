"""KAN-19 monitoring tests: webhook sender, job-failure notify CLI, heartbeat, run logs.

No live calls and no real webhook: `httpx.post` is replaced by a recorder, the DB is the SQLite
double from test_collector, and the subprocess tests point the webhook at a closed local port.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta, timezone

import httpx
import psycopg
import pytest

from pm25_forecast.monitoring import alerts, heartbeat, logs, notify
from pm25_forecast.storage import sql
from test_collector import DB_PASSWORD, NOW, FakeDb, do_run, settings

# Dummy webhook in the repo's existing pattern (hooks.example is reserved, never resolves).
WEBHOOK = "https://hooks.example/secret-token"
URL_PART = "secret-token"
GH_ENV = {
    "GITHUB_WORKFLOW": "Collector",
    "GITHUB_JOB": "collect",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "owner/pm25-forecast",
    "GITHUB_RUN_ID": "123456",
    "GITHUB_RUN_ATTEMPT": "2",
}


class Recorder:
    """Stands in for `httpx.post`: records calls, answers with `status` or raises `error`."""

    def __init__(self, status=200, error=None):
        self.calls: list[dict] = []
        self.status = status
        self.error = error

    def __call__(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, request=httpx.Request("POST", url))


@pytest.fixture
def fake_db():
    return FakeDb()


@pytest.fixture
def post(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(httpx, "post", recorder)
    return recorder


# --- sender --------------------------------------------------------------------------------------


def test_payload_serves_telegram_and_discord_and_is_bounded():
    assert alerts.payload("hi") == {"text": "hi", "content": "hi"}
    long = alerts.payload("x" * 5000)
    assert len(long["text"]) == alerts.MAX_CHARS
    assert long["text"].endswith("...")


def test_alert_posts_prefixed_message(post):
    alerts.make_alert(WEBHOOK, prefix="[p]")("boom")
    assert post.calls == [
        {"url": WEBHOOK, "json": {"text": "[p] boom", "content": "[p] boom"}, "timeout": 10.0}
    ]


def test_missing_webhook_logs_a_warning_and_does_not_raise(post, caplog):
    with caplog.at_level(logging.WARNING):
        alerts.make_alert(None)("boom")
        alerts.make_alert("")("boom")
    assert post.calls == []
    assert "ALERT: boom" in caplog.text
    assert "ALERT_WEBHOOK_URL is not set" in caplog.text


@pytest.mark.parametrize(
    "recorder",
    [
        Recorder(status=403),
        Recorder(error=httpx.ConnectError(f"cannot reach {WEBHOOK}")),
        Recorder(error=httpx.InvalidURL(f"bad url {WEBHOOK}")),
    ],
)
def test_delivery_failure_never_carries_the_url(monkeypatch, caplog, recorder):
    monkeypatch.setattr(httpx, "post", recorder)
    with pytest.raises(alerts.WebhookError) as info:
        alerts.post_message(WEBHOOK, "boom")
    assert URL_PART not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ or info.value.__context__ is None  # nothing chained
    with caplog.at_level(logging.DEBUG):
        assert alerts.safe_alert(alerts.make_alert(WEBHOOK), "boom") is False
    assert "alert failed: boom" in caplog.text
    assert URL_PART not in caplog.text


def test_safe_alert_reports_success():
    assert alerts.safe_alert(lambda _m: None, "x") is True
    assert alerts.safe_alert(None, "x") is True


# --- job-failure notify --------------------------------------------------------------------------


def test_failure_message_has_workflow_job_run_url_and_utc_time():
    bangkok = timezone(timedelta(hours=7))
    message = notify.failure_message(GH_ENV, datetime(2026, 10, 29, 8, 5, 9, tzinfo=bangkok))
    assert message == (
        "[pm25 ci] FAILURE: workflow Collector, job collect, at 2026-10-29T01:05:09Z, "
        "run https://github.com/owner/pm25-forecast/actions/runs/123456/attempts/2"
    )


def test_failure_message_outside_actions_and_with_odd_values():
    message = notify.failure_message(
        {"GITHUB_WORKFLOW": "Col\nlector", "GITHUB_RUN_ID": "abc"}, NOW
    )
    assert "workflow Col lector" in message
    assert "job ?" in message
    assert "run n/a (not in GitHub Actions)" in message
    assert notify.run_url({**GH_ENV, "GITHUB_RUN_ATTEMPT": ""}).endswith("/runs/123456")
    cancelled = notify.failure_message({**GH_ENV, "JOB_STATUS": "cancelled"}, NOW)
    assert cancelled.startswith("[pm25 ci] CANCELLED: workflow Collector")


def test_notify_posts_once_and_never_prints_the_url(post, caplog, capsys):
    env = {**GH_ENV, "ALERT_WEBHOOK_URL": f" {WEBHOOK} "}
    with caplog.at_level(logging.DEBUG):
        assert notify.main(["failure"], env=env, now=NOW) == notify.EXIT_OK
    assert len(post.calls) == 1
    assert post.calls[0]["url"] == WEBHOOK
    assert "actions/runs/123456" in post.calls[0]["json"]["text"]
    out = capsys.readouterr()
    assert URL_PART not in caplog.text + out.out + out.err


def test_notify_without_webhook_warns_and_exits_zero(post, caplog):
    with caplog.at_level(logging.WARNING):
        assert notify.main(["failure"], env=GH_ENV, now=NOW) == notify.EXIT_OK
    assert post.calls == []
    assert "ALERT_WEBHOOK_URL is not set" in caplog.text


def test_notify_delivery_failure_exits_one_without_the_url(monkeypatch, caplog, capsys):
    monkeypatch.setattr(httpx, "post", Recorder(error=httpx.ConnectError(f"x {WEBHOOK}")))
    with caplog.at_level(logging.DEBUG):
        code = notify.main(["failure"], env={**GH_ENV, "ALERT_WEBHOOK_URL": WEBHOOK}, now=NOW)
    assert code == notify.EXIT_SEND_FAILED
    out = capsys.readouterr()
    assert URL_PART not in caplog.text + out.out + out.err


def test_notify_rejects_unknown_kind():
    with pytest.raises(SystemExit):
        notify.main(["bogus"], env=GH_ENV, now=NOW)


def _run_module(module: str, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.lower().endswith("_proxy") and not k.startswith(("GITHUB_", "ALERT_", "DATABASE_"))
    }
    env.update(GH_ENV, **extra_env)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_notify_cli_end_to_end_logs_utc_and_hides_the_url():
    # Nothing listens on port 1: the post fails at once, exactly like a dead webhook host.
    proc = _run_module(
        "pm25_forecast.monitoring.notify",
        "failure",
        ALERT_WEBHOOK_URL="http://127.0.0.1:1/bot-" + URL_PART + "/sendMessage",
    )
    assert proc.returncode == notify.EXIT_SEND_FAILED
    assert URL_PART not in proc.stdout + proc.stderr
    assert "could not deliver the job-failure alert: alert webhook failed: ConnectError" in (
        proc.stderr
    )
    assert re.search(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z ERROR ", proc.stderr, re.M)


def test_notify_cli_without_webhook_exits_zero():
    proc = _run_module("pm25_forecast.monitoring.notify", "failure")
    assert proc.returncode == 0, proc.stderr
    assert "ALERT_WEBHOOK_URL is not set" in proc.stderr


# --- heartbeat -----------------------------------------------------------------------------------


def seed(fake_db, rows):
    sql.write_rows(
        fake_db,
        sql.UPSERT_OBSERVATION,
        [{"station_id": s, "ts": ts, "pm25": 10.0, "source": "air4thai"} for s, ts in rows],
    )
    fake_db.con.commit()


def hb_settings(tmp_path, **kw):
    return settings(tmp_path, station_locations="TST01=13.75,100.5;TST02=14,101", **kw)


def test_read_status_counts_only_the_last_24_hours(fake_db):
    seed(
        fake_db,
        [
            ("TST01", NOW - timedelta(hours=30)),  # outside the window
            ("TST01", NOW - timedelta(hours=2)),
            ("TST01", NOW - timedelta(hours=1)),
            ("TST02", NOW - timedelta(hours=5)),
        ],
    )
    status = heartbeat.read_status(fake_db, NOW, ["TST01", "TST02", "TST03"])
    assert status.rows_24h == 3
    assert status.last_ts == {"TST01": NOW - timedelta(hours=1), "TST02": NOW - timedelta(hours=5)}
    assert status.stale == ["TST02", "TST03"]


def test_format_message_levels():
    ok = heartbeat.Status(NOW, {"TST01": NOW - timedelta(minutes=30)}, 48, ("TST01",))
    assert heartbeat.format_message(ok) == (
        "OK 2026-10-26T07:30Z: 48 observations in the last 24 h; "
        "last ts: TST01 2026-10-26T07:00Z (0.5 h ago)"
    )
    warn = heartbeat.Status(NOW, {"TST01": NOW - timedelta(minutes=30)}, 48, ("TST01", "TST02"))
    assert heartbeat.format_message(warn).startswith("WARN ")
    assert "TST02 never (STALE)" in heartbeat.format_message(warn)
    dead = heartbeat.Status(NOW, {}, 0)
    assert heartbeat.format_message(dead) == (
        "FAIL 2026-10-26T07:30Z: 0 observations in the last 24 h; last ts: no stations"
    )


def test_heartbeat_posts_one_summary_line(tmp_path, fake_db, post, caplog):
    seed(fake_db, [("TST01", NOW - timedelta(hours=1)), ("TST02", NOW - timedelta(hours=1))])
    with caplog.at_level(logging.INFO):
        code = heartbeat.main(
            hb_settings(tmp_path, alert_webhook_url=WEBHOOK), connect=fake_db, clock=lambda: NOW
        )
    assert code == heartbeat.EXIT_OK
    assert len(post.calls) == 1
    text = post.calls[0]["json"]["text"]
    assert text.startswith("[pm25 heartbeat] OK 2026-10-26T07:30Z: 2 observations")
    assert "\n" not in text
    summary = _summary(caplog.text)
    assert summary["job"] == "heartbeat"
    assert summary["status"] == "ok"
    assert summary["rows_24h"] == 2
    assert URL_PART not in caplog.text


def test_heartbeat_without_webhook_warns_and_exits_zero(tmp_path, fake_db, post, caplog):
    seed(fake_db, [("TST01", NOW - timedelta(hours=1))])
    with caplog.at_level(logging.INFO):
        code = heartbeat.main(hb_settings(tmp_path), connect=fake_db, clock=lambda: NOW)
    assert code == heartbeat.EXIT_OK
    assert post.calls == []
    assert "ALERT_WEBHOOK_URL is not set" in caplog.text
    assert "HEARTBEAT: WARN" in caplog.text  # TST02 has never reported


def test_heartbeat_with_no_recent_rows_alerts_and_exits_one(tmp_path, fake_db, post):
    seed(fake_db, [("TST01", NOW - timedelta(days=3))])
    code = heartbeat.main(
        hb_settings(tmp_path, alert_webhook_url=WEBHOOK), connect=fake_db, clock=lambda: NOW
    )
    assert code == heartbeat.EXIT_UNHEALTHY
    assert len(post.calls) == 1
    assert post.calls[0]["json"]["text"].startswith("[pm25 heartbeat] FAIL ")


def test_heartbeat_db_down_alerts_instead_without_secrets(tmp_path, post, caplog):
    def down(url):
        raise psycopg.OperationalError(f"connection to {url} refused")

    with caplog.at_level(logging.DEBUG):
        code = heartbeat.main(
            hb_settings(tmp_path, alert_webhook_url=WEBHOOK), connect=down, clock=lambda: NOW
        )
    assert code == 1
    assert [c["json"]["text"] for c in post.calls] == [
        "[pm25 heartbeat] run skipped: database unreachable (OperationalError)"
    ]
    assert _summary(caplog.text)["status"] == "db_unreachable"
    assert DB_PASSWORD not in caplog.text
    assert URL_PART not in caplog.text


def test_heartbeat_db_down_with_the_real_driver(tmp_path, post, monkeypatch):
    from tenacity import wait_none

    from pm25_forecast.storage import db

    monkeypatch.setattr(db, "RETRY_WAIT", wait_none())
    code = heartbeat.main(hb_settings(tmp_path, alert_webhook_url=WEBHOOK), clock=lambda: NOW)
    assert code == 1
    assert len(post.calls) == 1
    assert "database unreachable" in post.calls[0]["json"]["text"]
    assert DB_PASSWORD not in post.calls[0]["json"]["text"]


def test_heartbeat_missing_database_url_is_a_config_alert(tmp_path, post):
    code = heartbeat.main(
        hb_settings(tmp_path, database_url=None, alert_webhook_url=WEBHOOK), clock=lambda: NOW
    )
    assert code == heartbeat.EXIT_CONFIG
    assert post.calls[0]["json"]["text"] == (
        "[pm25 heartbeat] configuration error: DATABASE_URL is not set"
    )


def test_heartbeat_crash_alerts_type_only(tmp_path, post):
    class Broken(FakeDb):
        def execute(self, statement, params=()):
            raise KeyError("detail that must not be alerted")

    code = heartbeat.main(
        hb_settings(tmp_path, alert_webhook_url=WEBHOOK), connect=Broken(), clock=lambda: NOW
    )
    assert code == heartbeat.EXIT_UNHEALTHY
    assert post.calls[0]["json"]["text"] == "[pm25 heartbeat] heartbeat crashed: KeyError"


def test_heartbeat_delivery_failure_exits_one_without_the_url(
    tmp_path, fake_db, monkeypatch, caplog
):
    seed(fake_db, [("TST01", NOW - timedelta(hours=1)), ("TST02", NOW - timedelta(hours=1))])
    monkeypatch.setattr(httpx, "post", Recorder(status=500))
    with caplog.at_level(logging.DEBUG):
        code = heartbeat.main(
            hb_settings(tmp_path, alert_webhook_url=WEBHOOK), connect=fake_db, clock=lambda: NOW
        )
    assert code == heartbeat.EXIT_UNHEALTHY
    assert "could not deliver the heartbeat: alert webhook returned HTTP 500" in caplog.text
    assert URL_PART not in caplog.text


def test_heartbeat_ignores_malformed_station_locations(tmp_path, fake_db, post):
    seed(fake_db, [("TST01", NOW - timedelta(hours=1))])
    code = heartbeat.main(
        settings(tmp_path, station_locations="garbage", alert_webhook_url=WEBHOOK),
        connect=fake_db,
        clock=lambda: NOW,
    )
    assert code == heartbeat.EXIT_OK
    assert "TST01" in post.calls[0]["json"]["text"]


# --- run logs ------------------------------------------------------------------------------------


def _summary(text: str) -> dict:
    lines = [line for line in text.splitlines() if logs.SUMMARY_TAG in line]
    assert lines, "no RUN_SUMMARY line logged"
    return json.loads(lines[-1].split(logs.SUMMARY_TAG, 1)[1])


def test_utc_formatter_ignores_the_host_timezone():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    record.created = 1_793_000_000.25
    record.msecs = 250.0
    line = logs.UtcFormatter(logs.LOG_FORMAT, logs.DATE_FORMAT).format(record)
    expected = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
    assert line == f"{expected}.250Z INFO x: hello"
    assert expected == datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S")


def test_format_summary_is_one_json_line():
    line = logs.format_summary("collector", {"started_at": NOW, "n": 1, "x": {1, 2} - {1, 2}})
    assert line.startswith("RUN_SUMMARY {")
    data = json.loads(line.removeprefix("RUN_SUMMARY "))
    assert data == {
        "job": "collector",
        "started_at": "2026-10-26T07:30:00+00:00",
        "n": 1,
        "x": "set()",
    }


def test_collector_run_logs_a_structured_summary(tmp_path, fake_db, caplog):
    with caplog.at_level(logging.INFO, logger="pm25_forecast.collector"):
        do_run(tmp_path, fake_db)
    summary = _summary(caplog.text)
    assert summary["job"] == "collector"
    assert summary["status"] == "ok"
    assert summary["started_at"] == NOW.isoformat()
    assert summary["observations_upserted"] == 1
    assert summary["failures"] == []
    assert DB_PASSWORD not in caplog.text
