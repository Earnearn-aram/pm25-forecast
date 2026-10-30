"""KAN-20: score matured forecasts against actuals. Offline: SQLite built from schema.sql."""

import argparse
import math
import socket
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from tenacity import wait_none

from pm25_forecast import score
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models.baselines import CAMS_PASSTHROUGH_MODEL_VERSION, PERSISTENCE_MODEL_VERSION
from pm25_forecast.score import ScoreResult, ScoreRow, rolling_metrics, run
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from test_predict import DB_PASSWORD, DB_URL, FakeDb

CHALLENGER = "kan15-retrospective-24h-lag1h"
NOW = datetime(2026, 10, 30, 12, 20, tzinfo=UTC)
H = timedelta(hours=1)
P = PERSISTENCE_MODEL_VERSION


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(db, "RETRY_WAIT", wait_none())


def settings(**kw):
    values = {"database_url": DB_URL, "challenger_model_version": CHALLENGER}
    return Settings(_env_file=None, **{**values, **kw})


def do_run(fake, now=NOW, since=None, **kw) -> ScoreResult:
    return run(settings(**kw), connect=fake, clock=lambda: now, since=since)


def add_obs(fake, station, ts, pm25):
    fake.con.execute(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, ?, 'test') "
        "ON CONFLICT (station_id, ts) DO UPDATE SET pm25 = excluded.pm25",
        (station, ts.isoformat(), pm25),
    )
    fake.con.commit()


def add_pred(fake, station, issue, h, model, pred):
    fake.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        " pm25_pred) VALUES (?, ?, ?, ?, 'r', ?)",
        (station, issue.isoformat(), (issue + h * H).isoformat(), model, pred),
    )
    fake.con.commit()


def scores(fake):
    return fake.con.execute(
        "SELECT station_id, valid_time, model_version, pm25_obs, abs_error FROM scores "
        "ORDER BY 1, 2, 3"
    ).fetchall()


@pytest.fixture
def fake():
    return FakeDb()


def test_matured_forecast_is_scored_and_future_one_is_not(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, P, 40.0)  # valid 10-30 12:00 <= NOW: matured
    add_pred(fake, "59t", issue + 2 * H, 24, P, 40.0)  # valid 10-30 14:00 > NOW: not yet
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    result = do_run(fake)
    assert scores(fake) == [("59t", (issue + 24 * H).isoformat(), P, 30.0, 10.0)]
    assert result.matured[P] == 1 and result.written[P] == 1 and result.waiting[P] == 0


def test_missing_actual_is_waiting_not_an_error(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", issue + 24 * H, None)  # null actual: not usable
    result = do_run(fake)
    assert scores(fake) == []
    assert result.waiting[P] == 1 and not result.failures


def test_rerun_is_idempotent_and_corrected_actual_updates_the_score(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    assert do_run(fake).written[P] == 1
    assert do_run(fake).written[P] == 0  # nothing changed: nothing written
    add_obs(fake, "59t", issue + 24 * H, 35.0)  # late correction
    assert do_run(fake).written[P] == 1
    assert scores(fake) == [("59t", (issue + 24 * H).isoformat(), P, 35.0, 5.0)]


def test_actual_corrected_to_null_drops_the_score_from_metrics(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    assert do_run(fake).metrics[P][24].n == 1
    add_obs(fake, "59t", issue + 24 * H, None)
    result = do_run(fake)
    assert result.waiting[P] == 1 and 24 not in result.metrics.get(P, {})


def test_summary_reports_empty_horizons_as_zero(fake, caplog):
    caplog.set_level("INFO")
    do_run(fake)
    assert '"+24h": {"n": 0}' in caplog.text


def test_only_persistence_and_challenger_are_scored_not_cams(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    for model in (P, CHALLENGER, CAMS_PASSTHROUGH_MODEL_VERSION, CAMS_MODEL_VERSION):
        add_pred(fake, "59t", issue, 24, model, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    do_run(fake)
    assert {row[2] for row in scores(fake)} == {P, CHALLENGER}


def test_without_challenger_only_persistence_is_scored(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, CHALLENGER, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    result = do_run(fake, challenger_model_version=None)
    assert list(result.matured) == [P] and scores(fake) == []


def test_forecast_older_than_the_lookback_is_not_rescored(fake):
    issue = NOW - score.RESCORE_LOOKBACK - 30 * H
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    assert do_run(fake).matured[P] == 0


def test_since_rescoring_reaches_back_beyond_the_default_lookback(fake):
    issue = NOW - score.RESCORE_LOOKBACK - 30 * H
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    assert do_run(fake).matured[P] == 0
    since = issue + 24 * H - H
    assert do_run(fake, since=since).matured[P] == 1
    assert [r[-2:] for r in scores(fake)] == [(30.0, 10.0)]


def test_since_corrects_a_score_after_the_observation_was_repaired(fake):
    issue = NOW - score.RESCORE_LOOKBACK - 30 * H
    valid = issue + 24 * H
    add_pred(fake, "59t", issue, 24, P, 40.0)
    add_obs(fake, "59t", valid, 30.0)
    do_run(fake, since=valid - H)
    fake.con.execute("UPDATE observations SET pm25 = 38.0")
    do_run(fake, since=valid - H)
    (row,) = scores(fake)
    assert row[-2:] == (38.0, 2.0)


def test_parse_since_requires_an_offset():
    assert score.parse_since("2026-10-27T00:00:00Z") == datetime(2026, 10, 27, tzinfo=UTC)
    assert score.parse_since("2026-10-27T07:00:00+07:00") == datetime(2026, 10, 27, tzinfo=UTC)
    with pytest.raises(argparse.ArgumentTypeError):
        score.parse_since("2026-10-27T00:00:00")
    with pytest.raises(argparse.ArgumentTypeError):
        score.parse_since("yesterday")


def test_main_passes_since_to_run(monkeypatch):
    monkeypatch.setattr(score, "get_settings", lambda: settings())
    seen = {}

    def fake_run(_settings, **kw):
        seen.update(kw)
        return ScoreResult(run_id="r", scored_at=NOW)

    monkeypatch.setattr(score, "run", fake_run)
    assert score.main(["--since", "2026-10-27T00:00:00Z"]) == 0
    assert seen["since"] == datetime(2026, 10, 27, tzinfo=UTC)
    assert score.main([]) == 0
    assert seen["since"] is None


def test_end_to_end_metrics_and_skill(fake):
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, P, 40.0)  # |40-30| = 10
    add_pred(fake, "59t", issue, 24, CHALLENGER, 35.0)  # |35-30| = 5
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    m = do_run(fake).metrics
    assert m[P][24].mae == 10.0 and m[P][24].skill_vs_persistence is None
    assert m[CHALLENGER][24].mae == 5.0
    assert m[CHALLENGER][24].skill_vs_persistence == pytest.approx(0.5)


def _row(station, issue, h, err):
    return ScoreRow(station_id=station, issue_time=issue, valid_time=issue + h * H, abs_error=err)


def test_skill_uses_matched_pairs_only():
    t0 = datetime(2026, 10, 29, 0, tzinfo=UTC)
    persistence = [_row("59t", t0, 6, 4.0), _row("59t", t0 + H, 6, 100.0)]  # 2nd: no challenger
    challenger = [_row("59t", t0, 6, 2.0), _row("61t", t0, 6, 50.0)]  # 61t: no persistence
    m = rolling_metrics({P: persistence, CHALLENGER: challenger})
    c = m[CHALLENGER][6]
    assert c.n == 2 and c.mae == 26.0 and c.n_matched == 1
    assert c.skill_vs_persistence == pytest.approx(1 - 2.0 / 4.0)
    assert m[P][6].rmse == pytest.approx(math.sqrt((16 + 10000) / 2))


def test_metrics_are_split_by_horizon_and_ignore_other_horizons():
    t0 = datetime(2026, 10, 29, 0, tzinfo=UTC)
    rows = [_row("59t", t0, 6, 1.0), _row("59t", t0, 12, 2.0), _row("59t", t0, 3, 9.0)]
    m = rolling_metrics({P: rows})
    assert set(m[P]) == {6, 12} and m[P][12].mae == 2.0


def test_naive_clock_is_rejected(fake):
    with pytest.raises(ValueError):
        do_run(fake, now=NOW.replace(tzinfo=None))


def test_missing_database_url_is_a_config_error(fake):
    with pytest.raises(score.ConfigError):
        run(Settings(_env_file=None), connect=fake, clock=lambda: NOW)


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


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(score, "get_settings", lambda: settings())

    def use(fn):
        monkeypatch.setattr(score, "run", fn)

    use(lambda *_a, **_k: ScoreResult(run_id="r", scored_at=NOW))
    assert score.main() == 0
    use(lambda *_a, **_k: ScoreResult(run_id="r", scored_at=NOW, failures=["x"]))
    assert score.main() == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    use(down)
    assert score.main() == 1

    def bad_config(*_a, **_k):
        raise score.ConfigError("nope")

    use(bad_config)
    assert score.main() == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    use(crash)
    assert score.main() == 1
