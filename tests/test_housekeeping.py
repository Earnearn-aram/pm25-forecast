"""KAN-24: weekly housekeeping (predictions retention + dry-run registry prune). Offline."""

import json
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from pm25_forecast import housekeeping
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from test_predict import DB_URL, FakeDb

NOW = datetime(2026, 10, 30, 4, 17, tzinfo=UTC)
CUTOFF = NOW - timedelta(days=90)


class HousekeepingDb(FakeDb):
    """FakeDb that runs the Postgres-only retention SQL as its SQLite equivalent."""

    def execute(self, statement, params=()):
        if statement == sql.COUNT_OLD_PREDICTIONS:
            return self.con.execute(
                "SELECT count(*) FROM predictions WHERE issue_time < ?", (CUTOFF.isoformat(),)
            )
        if statement == sql.PURGE_OLD_PREDICTIONS:
            return self.con.execute(
                "DELETE FROM predictions WHERE issue_time < ?", (CUTOFF.isoformat(),)
            )
        return super().execute(statement, params)


def settings(**kw):
    return Settings(_env_file=None, database_url=DB_URL, **kw)


def seed(fake):
    for i, age in enumerate((120, 91, 89, 1)):
        issue = (NOW - timedelta(days=age)).isoformat()
        fake.con.execute(
            "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
            " pm25_pred) VALUES ('59t', ?, ?, 'cams_global', ?, 1.0)",
            (issue, issue, f"r{i}"),
        )
    for i in range(1, 7):  # m6 is the newest
        fake.con.execute(
            "INSERT INTO models (model_version, sha256, artifact, manifest, created_at) "
            "VALUES (?, 'x', x'00', '{}', ?)",
            (f"m{i}", (NOW - timedelta(days=10 - i)).isoformat()),
        )
    fake.con.execute("INSERT INTO model_champion (slot, model_version) VALUES ('production', 'm1')")
    fake.con.commit()


@pytest.fixture
def fake():
    fake = HousekeepingDb()
    seed(fake)
    return fake


def count(fake, table):
    return fake.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_retention_sql_uses_the_90_day_rule_on_both_statements():
    for statement in (sql.COUNT_OLD_PREDICTIONS, sql.PURGE_OLD_PREDICTIONS):
        assert "issue_time < now() - interval '90 days'" in statement


def test_run_counts_then_purges_old_predictions(fake):
    result = housekeeping.run(settings(), connect=fake, clock=lambda: NOW)
    assert result.predictions_due == 2 and result.predictions_deleted == 2
    assert count(fake, "predictions") == 2  # the 89 and 1 day old rows stay


def test_dry_run_counts_but_deletes_nothing(fake):
    result = housekeeping.run(settings(), connect=fake, clock=lambda: NOW, dry_run=True)
    assert result.predictions_due == 2 and result.predictions_deleted == 0
    assert count(fake, "predictions") == 4


def test_prune_is_only_ever_a_dry_run_here(fake):
    result = housekeeping.run(
        settings(challenger_model_version="m2"), connect=fake, clock=lambda: NOW
    )
    # newest 4 are m3..m6; champion m1 and challenger m2 are protected too: nothing to delete
    assert result.prune_would_delete == []
    assert result.prune_keep["m1"] == "production champion"
    no_challenger = settings(challenger_model_version=None)
    lone = housekeeping.run(no_challenger, connect=fake, clock=lambda: NOW)
    assert lone.prune_would_delete == ["m2"]
    assert count(fake, "models") == 6  # nothing was deleted


def test_summary_line_holds_both_results(fake, caplog):
    with caplog.at_level("INFO"):
        housekeeping.run(settings(), connect=fake, clock=lambda: NOW)
    lines = [r.getMessage() for r in caplog.records if "RUN_SUMMARY" in r.getMessage()]
    assert len(lines) == 1
    payload = json.loads(lines[0].split("RUN_SUMMARY ", 1)[1])
    assert payload["job"] == "housekeeping" and payload["prune_mode"] == "dry_run"
    assert payload["predictions_due"] == 2 and payload["predictions_deleted"] == 2
    assert payload["prune_would_delete"] == ["m2"]


def test_missing_database_url_is_a_config_error():
    with pytest.raises(housekeeping.ConfigError):
        housekeeping.run(Settings(_env_file=None), connect=FakeDb())


def test_db_down_skips_alerts_and_exits_nonzero():
    def refuse(_url):
        raise psycopg.OperationalError("connection refused")

    alerts: list[str] = []
    result, code = run_entrypoint(
        lambda: housekeeping.run(settings(), connect=refuse, clock=lambda: NOW),
        alerts.append,
        connection_errors=db.CONNECTION_ERRORS,
    )
    assert isinstance(result, RunSkipped) and code == 1 and len(alerts) == 1


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(housekeeping, "get_settings", lambda: settings())
    seen = {}

    def ok(_settings, **kw):
        seen.update(kw)
        return housekeeping.HousekeepingResult(run_id="r")

    monkeypatch.setattr(housekeeping, "run", ok)
    assert housekeeping.main(["--dry-run"]) == 0 and seen["dry_run"] is True

    def bad_config(*_a, **_k):
        raise housekeeping.ConfigError("nope")

    monkeypatch.setattr(housekeeping, "run", bad_config)
    assert housekeeping.main([]) == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(housekeeping, "run", crash)
    assert housekeeping.main([]) == 1
