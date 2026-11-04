pyenv: cannot rehash: /Users/earn/.pyenv/shims isn't writable
"""KAN-23: weekly retrain with champion/challenger comparison. Offline, synthetic data only."""

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import psycopg
import pytest
from pydantic import ValidationError

from pm25_forecast import retrain
from pm25_forecast.retrain import Decision, Evaluation, decide, retrain_version
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from pm25_forecast.storage.registry import (
    artifact_sha256,
    promote_model,
    read_model,
    write_model,
)
from pm25_forecast.train_model import build_samples, train

START = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 2, 10, 0, 30, tzinfo=UTC)  # holdout_start = 2026-02-03 00:00 UTC
HOLDOUT_START = datetime(2026, 2, 3, tzinfo=UTC)
DB_URL = "postgresql://user@127.0.0.1:1/db"  # no password, nothing secret


class Registry:
    """In-memory stand-in for the psycopg connection used by storage/registry.py."""

    def __init__(self):
        self.rows = {}
        self.champion = None
        self.events = []  # model_events rows (parameter dicts), oldest first
        self.last_retrain_at = None  # stand-in for max(at) over retrain/auto events
        self.lock = threading.Lock()
        self._selected = None

    def __call__(self, _url):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, params):
        with self.lock:
            if statement.lstrip().startswith("INSERT INTO model_events"):
                self.events.append(params)
                if params["actor"] in ("retrain", "auto"):
                    self.last_retrain_at = datetime.now(UTC)
            elif "max(at) FROM model_events" in statement:
                self._selected = (self.last_retrain_at,)
            elif "FROM model_events" in statement:
                changes = [e for e in self.events if e["action"] in ("promote", "rollback")]
                self._selected = None
                if changes:
                    self._selected = (changes[-1]["model_version"], changes[-1]["previous_version"])
            elif statement.lstrip().startswith("INSERT INTO models"):
                self.rows.setdefault(
                    params["model_version"],
                    (params["sha256"], params["artifact"], params["manifest"]),
                )
            elif statement.lstrip().startswith("INSERT INTO model_champion"):
                self.champion = params["model_version"]
            elif "FROM model_champion" in statement:
                self._selected = (self.champion,) if self.champion else None
            elif statement.lstrip().startswith("SELECT"):
                self._selected = self.rows.get(params["model_version"])
        return self

    def fetchone(self):
        with self.lock:
            return self._selected


def write_export(path: Path, hours: int = 960) -> None:
    """Level alternates between 10 and 40 each UTC day, so persistence at +24 h is always wrong
    by 30 and a model that reads the latest value learns the flip."""
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE observations "
            "(station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, source VARCHAR, "
            "available_at TIMESTAMPTZ)"
        )
        con.execute(
            "CREATE OR REPLACE TABLE forecast_vintages "
            "(station_id VARCHAR, issue_time TIMESTAMPTZ, valid_time TIMESTAMPTZ, "
            "model_version VARCHAR, run_id VARCHAR, pm25_pred DOUBLE, available_at TIMESTAMPTZ)"
        )
        rows = [
            (
                "59t",
                START + timedelta(hours=i),
                10.0 + 30.0 * ((i // 24) % 2),
                "test",
                START + timedelta(hours=i),
            )
            for i in range(hours)
        ]
        con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", rows)


def register_variant(reg, source, name, *, horizons=None, manifest_update=None):
    """Register a copy of an existing bundle under a new name (optionally fewer horizons)."""
    artifact, manifest = read_model(reg, source)
    if horizons is not None:
        bundle = json.loads(artifact)
        bundle["horizons"] = {h: t for h, t in bundle["horizons"].items() if h in horizons}
        artifact = json.dumps(bundle, separators=(",", ":")).encode()
    manifest = {
        **manifest,
        "version": name,
        "artifact_sha256": artifact_sha256(artifact),
        **(manifest_update or {}),
    }
    write_model(reg, name, artifact, manifest)


def export_fake(path, _url):
    write_export(path)


def settings(**kw) -> Settings:
    """Explicit defaults so a developer's .env or environment cannot change the tests."""
    values = {"challenger_model_version": None, "auto_promote": False, **kw}
    return Settings(database_url=DB_URL, **values)


def do_run(reg, alerts, *, now=NOW, export=export_fake, **kw):
    s = kw.pop("settings", None) or settings()
    # Existing downstream decision tests simulate a future verified-boundary caller. The
    # production entrypoint itself fails closed; dedicated tests below assert that behavior.
    with patch.object(retrain, "_trusted_holdout_boundary_verified", return_value=True):
        return retrain.run(
            s, connect=reg, clock=lambda: now, export=export, alert=alerts.append, **kw
        )


# --- decide: boundaries ------------------------------------------------------------------------


def ev(version="m", n=100, mae=10.0, persistence=20.0):
    return Evaluation(version, n, mae, mae * 1.2, persistence)


def test_promote_at_exactly_min_pairs_and_exactly_two_percent_gain():
    decision = decide(ev("cand", n=100, mae=9.8), ev("champ", n=100, mae=10.0))
    assert isinstance(decision, Decision)
    assert decision.action == "promote"
    assert decision.metrics["n_pairs"] == 100


def test_keep_one_pair_short_of_min_pairs():
    decision = decide(ev("cand", n=99, mae=5.0), ev("champ", n=99, mae=10.0))
    assert decision.action == "keep"
    assert "99" in decision.reason


def test_keep_just_below_two_percent_gain():
    decision = decide(ev("cand", mae=9.81), ev("champ", mae=10.0))
    assert decision.action == "keep"
    assert "below 2%" in decision.reason


def test_keep_when_candidate_is_worse_or_equal_to_incumbent():
    assert decide(ev("cand", mae=10.0), ev("champ", mae=10.0)).action == "keep"
    assert decide(ev("cand", mae=11.0), ev("champ", mae=10.0)).action == "keep"


def test_keep_when_candidate_does_not_beat_persistence_even_if_it_beats_the_champion():
    assert decide(ev("cand", mae=9.0, persistence=9.0), ev("champ", mae=30.0)).action == "keep"
    assert decide(ev("cand", mae=9.0, persistence=8.0), ev("champ", mae=30.0)).action == "keep"
    assert decide(ev("cand", mae=9.0, persistence=9.01), ev("champ", mae=30.0)).action == "promote"


def test_keep_without_a_persistence_reference_or_with_bad_numbers():
    assert decide(ev("cand", mae=1.0, persistence=None), None).action == "keep"
    assert decide(ev("cand", mae=float("nan")), None).action == "keep"


def test_no_champion_promotes_only_if_pairs_and_persistence_checks_pass():
    assert decide(ev("cand", mae=9.0, persistence=20.0), None).action == "promote"
    assert decide(ev("cand", n=99, mae=9.0), None).action == "keep"


def test_custom_thresholds_are_honoured():
    cand, champ = ev("cand", n=50, mae=9.0), ev("champ", n=50, mae=10.0)
    assert decide(cand, champ).action == "keep"
    assert decide(cand, champ, min_pairs=50, min_rel_gain=0.10).action == "promote"
    assert decide(cand, champ, min_pairs=50, min_rel_gain=0.11).action == "keep"


def test_keep_when_incumbent_mae_is_zero():
    assert decide(ev("cand", mae=0.0), ev("champ", mae=0.0)).action == "keep"


def test_keep_when_candidate_regresses_at_another_horizon_even_if_24h_wins():
    cand, champ = ev("cand", mae=5.0), ev("champ", mae=10.0)
    worse = [(6, ev("cand", mae=10.21), ev("champ", mae=10.0))]
    decision = decide(cand, champ, other_horizons=worse)
    assert decision.action == "keep"
    assert "+6h" in decision.reason
    # exactly 2% worse is not "more than 2%"; better at the other horizon is fine
    edge = [(6, ev("cand", mae=10.2), ev("champ", mae=10.0))]
    assert decide(cand, champ, other_horizons=edge).action == "promote"
    better = [(12, ev("cand", mae=8.0), ev("champ", mae=10.0))]
    assert decide(cand, champ, other_horizons=better).action == "promote"


# --- version naming and the training window ---------------------------------------------------


def test_retrain_version_uses_utc_date_and_suffixes_same_day_reruns():
    taken: set[str] = set()
    now = datetime(2026, 2, 10, 23, 59, tzinfo=UTC)
    assert retrain_version(now, taken.__contains__) == "retrain-20260210"
    taken.add("retrain-20260210")
    assert retrain_version(now, taken.__contains__) == "retrain-20260210-2"
    taken.add("retrain-20260210-2")
    assert retrain_version(now, taken.__contains__) == "retrain-20260210-3"


def test_training_export_keeps_only_rows_before_the_holdout(tmp_path):
    full, cut = tmp_path / "full.duckdb", tmp_path / "cut.duckdb"
    write_export(full)
    retrain.make_training_export(full, cut, HOLDOUT_START)
    with duckdb.connect(str(cut), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        newest, count = con.execute("SELECT max(ts), count(*) FROM observations").fetchone()
    assert newest == HOLDOUT_START - timedelta(hours=1)
    assert count == 33 * 24


def test_training_export_drops_rows_that_arrived_after_the_cutoff(tmp_path):
    full, cut = tmp_path / "full.duckdb", tmp_path / "cut.duckdb"
    write_export(full)
    late_label = HOLDOUT_START - timedelta(hours=2)  # dated before the cutoff, ingested after it
    issue = HOLDOUT_START - timedelta(hours=30)
    with duckdb.connect(str(full)) as con:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(
            "UPDATE observations SET available_at = ? WHERE ts = ?",
            [HOLDOUT_START + timedelta(hours=5), late_label],
        )
        con.execute(
            "INSERT INTO forecast_vintages VALUES "
            "('59t', ?, ?, 'cams_global', 'r', 20.0, ?), "
            "('59t', ?, ?, 'cams_global', 'r', 21.0, ?)",
            [
                issue,
                issue + timedelta(hours=24),
                issue + timedelta(hours=1),  # known in time: kept
                issue,
                issue + timedelta(hours=25),
                HOLDOUT_START + timedelta(hours=1),  # ingested late: dropped
            ],
        )
    retrain.make_training_export(full, cut, HOLDOUT_START)
    with duckdb.connect(str(cut), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        late = con.execute("SELECT count(*) FROM observations WHERE ts = ?", [late_label])
        assert late.fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM observations").fetchone()[0] == 33 * 24 - 1
        vintages = con.execute("SELECT pm25_pred FROM forecast_vintages").fetchall()
    assert vintages == [(20.0,)]


def test_build_samples_as_of_ignores_labels_after_the_cut(tmp_path):
    export = tmp_path / "history.duckdb"
    write_export(export, hours=96)
    as_of = START + timedelta(hours=60)

    cut, _labels, _vintages = build_samples(export, as_of)
    for horizon, rows in cut.items():
        assert rows
        for _x, _y, issue, label_available, _station in rows:
            assert issue + timedelta(hours=horizon) <= as_of and label_available <= as_of
    assert len(build_samples(export)[0][24]) > len(cut[24])


# --- the run ----------------------------------------------------------------------------------


def test_run_registers_candidate_and_only_recommends_when_auto_promote_is_off(caplog):
    reg, alerts = Registry(), []
    with caplog.at_level("INFO"):
        result = do_run(reg, alerts)

    assert result.version == "retrain-20260210"
    assert result.status == "ok" and not result.failures
    assert result.version in reg.rows  # always registered
    assert reg.champion is None  # production untouched
    assert result.decision.action == "promote"
    assert result.recommended and not result.promoted
    assert any(a.startswith(f"recommend promote {result.version}") for a in alerts)

    table = result.comparison["+24h"]
    assert table["n_pairs"] >= 100
    assert set(table["models"]) == {result.version, "persistence-24h"}
    assert result.artifact_bytes == len(read_model(reg, result.version)[0])
    assert table["models"][result.version]["mae"] < table["models"]["persistence-24h"]["mae"]
    assert set(result.comparison) == {"+6h", "+12h", "+24h"}

    summary = [r.getMessage() for r in caplog.records if "RUN_SUMMARY" in r.getMessage()]
    assert len(summary) == 1
    payload = json.loads(summary[0].split("RUN_SUMMARY ", 1)[1])
    assert payload["job"] == "retrain" and payload["auto_promote"] is False
    assert payload["decision"]["action"] == "promote"
    assert payload["candidate_artifact_bytes"] == result.artifact_bytes > 0
    assert "+24h" in payload["comparison"]


def test_training_window_ends_before_the_holdout_window():
    reg, seen = Registry(), {}

    def spy(path, version, as_of=None):
        seen["as_of"] = as_of
        return train(path, version, as_of=as_of)

    result = do_run(reg, [], trainer=spy)
    assert seen["as_of"] == HOLDOUT_START  # the training knowledge cut is the holdout start
    _artifact, manifest = read_model(reg, result.version)
    assert datetime.fromisoformat(manifest["training_window"]["end"]) < HOLDOUT_START
    assert manifest["holdout_start"] == HOLDOUT_START.isoformat()


def test_auto_promote_true_is_blocked_before_data_or_registry_access():
    def forbidden(*_args, **_kwargs):
        raise AssertionError("automatic promotion must block before data or registry access")

    alerts = []
    with patch.object(retrain, "_trusted_holdout_boundary_verified", return_value=True):
        result = retrain.run(
            settings(auto_promote=True),
            connect=forbidden,
            export=forbidden,
            trainer=forbidden,
            clock=lambda: NOW,
            alert=alerts.append,
        )

    assert result.status == retrain.BLOCKED_AUTO_PROMOTE_STATUS
    assert result.warnings == [retrain.BLOCKED_AUTO_PROMOTE_REASON]
    assert not result.version and not result.promoted and not result.recommended
    assert alerts == []  # run returns a result; the CLI sends the reason-specific alert


def test_manual_promotion_uses_the_exact_evaluated_candidate_artifact(monkeypatch):
    reg, alerts = Registry(), []
    evaluated = {}
    real_publish = retrain.publish_model

    def capture_publish(url, version, artifact, manifest, **kwargs):
        evaluated[version] = artifact
        return real_publish(url, version, artifact, manifest, **kwargs)

    monkeypatch.setattr(retrain, "publish_model", capture_publish)
    result = do_run(reg, alerts)

    assert result.recommended and not result.promoted
    assert reg.champion is None
    stored_artifact, _manifest = read_model(reg, result.version)
    assert stored_artifact == evaluated[result.version]

    # The reviewed manual command promotes that exact immutable registry version.
    promote_model(reg, result.version, actor="cli", reason="owner reviewed candidate")
    assert reg.champion == result.version
    promote_events = [event for event in reg.events if event["action"] == "promote"]
    assert len(promote_events) == 1
    assert promote_events[0]["model_version"] == result.version
    assert promote_events[0]["actor"] == "cli"


def test_same_day_rerun_gets_suffix_and_is_compared_with_the_champion():
    reg = Registry()
    first = do_run(reg, [])
    reg.champion = first.version  # owner-approved manual promotion for the fixture
    second = do_run(reg, [], now=NOW + timedelta(hours=2))

    assert second.version == f"{first.version}-2"
    assert reg.champion == first.version  # auto off: pointer unchanged whatever the decision
    assert second.roles["champion"] == first.version
    models = second.comparison["+24h"]["models"]
    assert {first.version, second.version, "persistence-24h"} <= set(models)


def test_challenger_is_scored_when_set_and_skipped_when_unloadable():
    reg = Registry()
    first = do_run(reg, [])
    with_challenger = do_run(
        reg,
        [],
        now=NOW + timedelta(hours=1),
        settings=settings(challenger_model_version=first.version),
    )
    assert first.version in with_challenger.challenger_comparison["+24h"]["models"]
    assert with_challenger.roles["challenger"] == first.version

    missing = do_run(
        reg, [], now=NOW + timedelta(hours=2), settings=settings(challenger_model_version="nope")
    )
    assert missing.challenger_comparison == {}
    assert "challenger_error" in missing.roles
    assert missing.status == "ok"


def test_unloadable_champion_means_keep_and_a_failed_run():
    reg, alerts = Registry(), []
    first = do_run(reg, [])
    artifact, manifest = read_model(reg, first.version)
    bad = {**manifest, "version": "bad-champion", "assumed_reporting_lag_hours": 0}
    write_model(reg, "bad-champion", artifact, bad)
    reg.champion = "bad-champion"

    result = do_run(reg, alerts, now=NOW + timedelta(hours=1))
    assert result.decision.action == "keep"
    assert result.status == "failed" and result.failures
    assert reg.champion == "bad-champion"  # never promoted over a champion we could not score


def test_champion_that_cannot_be_scored_means_keep_never_promote():
    reg, alerts = Registry(), []
    first = do_run(reg, [])
    register_variant(reg, first.version, "partial-champion", horizons={"6"})  # no +24 h booster
    reg.champion = "partial-champion"

    result = do_run(reg, alerts, now=NOW + timedelta(hours=1))
    assert result.decision.action == "keep"
    assert "could not be scored" in result.decision.reason
    assert result.status == "failed"
    assert reg.champion == "partial-champion"
    assert not result.promoted and not result.recommended


def test_challenger_never_changes_the_decision_pairs():
    reg = Registry()
    first = do_run(reg, [])
    register_variant(reg, first.version, "partial-challenger", horizons={"6"})
    second = do_run(reg, [], settings=settings(challenger_model_version="partial-challenger"))

    assert second.comparison["+24h"]["n_pairs"] == first.comparison["+24h"]["n_pairs"]
    assert "partial-challenger" not in second.comparison["+6h"]["models"]
    assert "partial-challenger" in second.challenger_comparison["+6h"]["models"]


def test_overlap_with_the_holdout_is_flagged_in_summary_and_alert(caplog):
    reg, alerts = Registry(), []
    first = do_run(reg, [])
    window = {"start": "2026-01-01T00:00:00+00:00", "end": "2026-02-05T00:00:00+00:00"}
    overlapping = {"training_window": window}
    register_variant(reg, first.version, "overlapping", manifest_update=overlapping)

    with caplog.at_level("INFO"):
        result = do_run(reg, alerts, settings=settings(challenger_model_version="overlapping"))
    assert any("challenger overlapping overlaps holdout" in w for w in result.warnings)
    assert result.recommended
    assert any("recommend promote" in a and "overlaps holdout" in a for a in alerts)
    summary = [r.getMessage() for r in caplog.records if "RUN_SUMMARY" in r.getMessage()][-1]
    assert "overlaps holdout" in summary


def test_insufficient_history_skips_without_registering_or_promoting():
    reg, alerts = Registry(), []

    def tiny(path, _url):
        write_export(path, hours=30)

    result = do_run(reg, alerts, export=tiny)
    assert result.status == "skipped" and not result.failures
    assert result.warnings and "training data not ready" in result.warnings[0]
    assert (NOW.replace(minute=0) - timedelta(days=7)).isoformat() in result.warnings[0]
    assert reg.rows == {} and reg.champion is None


def test_unrelated_training_value_error_remains_failure():
    reg = Registry()

    def broken(*_args, **_kwargs):
        raise ValueError("invalid model configuration")

    result = do_run(reg, [], trainer=broken)
    assert result.status == "failed"
    assert "invalid model configuration" in result.failures[0]


def test_config_errors():
    reg = Registry()
    with pytest.raises(retrain.ConfigError):
        retrain.run(Settings(database_url=None), connect=reg, clock=lambda: NOW)
    with pytest.raises(retrain.ConfigError):
        retrain.run(settings(), connect=reg, clock=lambda: NOW, holdout_days=0)


@pytest.mark.parametrize("if_drift", [False, True])
@pytest.mark.parametrize("auto_promote", [False, True])
def test_unverified_holdout_boundary_blocks_before_any_data_or_registry_access(
    if_drift, auto_promote, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="pm25_forecast.retrain")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retraining must block before this operation")

    for name in (
        "build_samples",
        "drift_gate",
        "publish_model",
        "compare",
        "read_champion",
        "read_model",
        "record_event",
        "_registered",
    ):
        monkeypatch.setattr(retrain, name, forbidden)
    alerts = []
    result = retrain.run(
        settings(auto_promote=auto_promote),
        connect=forbidden,
        export=forbidden,
        trainer=forbidden,
        clock=lambda: NOW,
        alert=alerts.append,
        if_drift=if_drift,
    )
    assert result.status == retrain.BLOCKED_RETRAIN_STATUS
    assert result.warnings == [retrain.BLOCKED_RETRAIN_REASON]
    assert not result.version and not result.promoted and not result.recommended
    assert f'"status":"{retrain.BLOCKED_RETRAIN_STATUS}"' in caplog.text.replace(" ", "")
    assert alerts == []  # run returns a result; the CLI sends exactly one operational alert


@pytest.mark.parametrize("argv", [[], ["--if-drift"]])
def test_main_alerts_once_and_exits_nonzero_when_boundary_is_unverified(
    argv, monkeypatch, tmp_path
):
    monkeypatch.setattr(retrain, "get_settings", lambda: settings(auto_promote=True))
    alerts = []
    monkeypatch.setattr(retrain, "make_alert", lambda *_args, **_kwargs: alerts.append)
    output_file = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    assert retrain.main(argv) == retrain.EXIT_FAILED
    assert alerts == [retrain.BLOCKED_RETRAIN_REASON]
    assert output_file.read_text(encoding="utf-8") == "failure_alert_sent=true\n"


def test_main_alerts_once_when_auto_promotion_requires_owner_review(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "get_settings", lambda: settings(auto_promote=True))
    monkeypatch.setattr(retrain, "_trusted_holdout_boundary_verified", lambda: True)
    alerts = []
    monkeypatch.setattr(retrain, "make_alert", lambda *_args, **_kwargs: alerts.append)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    output_file = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    assert retrain.main([]) == retrain.EXIT_FAILED
    assert alerts == [retrain.BLOCKED_AUTO_PROMOTE_REASON]
    assert output_file.read_text(encoding="utf-8") == "failure_alert_sent=true\n"


def test_workflow_alert_marker_is_scoped_to_runner_output(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outside-output"))
    retrain._mark_workflow_alert_sent()
    assert not (tmp_path / "outside-output").exists()

    monkeypatch.delenv("GITHUB_OUTPUT")
    retrain._mark_workflow_alert_sent()  # Missing Actions output is a harmless no-op.


def test_failed_block_alert_leaves_generic_workflow_fallback_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "get_settings", lambda: settings())

    def fail_alert(*_args, **_kwargs):
        def fail(_message):
            raise RuntimeError("webhook unavailable")

        return fail

    monkeypatch.setattr(retrain, "make_alert", fail_alert)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    output_file = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    assert retrain.main([]) == retrain.EXIT_FAILED
    assert not output_file.exists()


def test_retrain_workflows_use_specific_alert_before_generic_fallback():
    root = Path(__file__).parents[1]
    workflow_paths = (
        root / ".github/workflows/retrain.yml",
        root / "docs/retrain-workflow.yml.txt",
        root / "docs/drift-workflow.yml.txt",
    )
    for path in workflow_paths:
        text = path.read_text(encoding="utf-8")
        assert "steps.retrain.outputs.failure_alert_sent != 'true'" in text
        assert "(failure() || cancelled())" in text


# --- AUTO_PROMOTE setting ---------------------------------------------------------------------


def test_auto_promote_defaults_off_and_parses_strictly(monkeypatch):
    monkeypatch.delenv("AUTO_PROMOTE", raising=False)
    assert Settings(_env_file=None).auto_promote is False
    assert Settings(auto_promote="").auto_promote is False  # unset repo variable
    assert Settings(auto_promote="false").auto_promote is False
    assert Settings(auto_promote="true").auto_promote is True
    assert Settings(auto_promote=" TRUE ").auto_promote is True
    assert Settings(auto_promote="False").auto_promote is False
    for bad in ("maybe", "yes", "1", "on", "t", "y"):
        with pytest.raises(ValidationError):
            Settings(auto_promote=bad)


# --- DB down and exit codes -------------------------------------------------------------------


def test_db_down_skips_alerts_and_exits_nonzero_without_leaking_the_url(caplog):
    def refuse(_url):
        raise psycopg.OperationalError(f"connection refused {DB_URL}")

    alerts: list[str] = []
    with patch.object(retrain, "_trusted_holdout_boundary_verified", return_value=True):
        result, code = run_entrypoint(
            lambda: retrain.run(settings(), connect=refuse, clock=lambda: NOW),
            alerts.append,
            connection_errors=db.CONNECTION_ERRORS,
        )
    assert isinstance(result, RunSkipped)
    assert code == 1
    assert len(alerts) == 1 and "OperationalError" in alerts[0]
    assert DB_URL not in alerts[0] and DB_URL not in caplog.text


@pytest.fixture
def patched_main(monkeypatch):
    monkeypatch.setattr(retrain, "get_settings", lambda: settings())

    def patch_run(fn):
        monkeypatch.setattr(retrain, "run", fn)

    return patch_run


def test_main_exit_codes(patched_main, monkeypatch):
    patched_main(lambda *_a, **_k: retrain.RetrainResult(run_id="r"))
    assert retrain.main([]) == 0

    alerts = []
    monkeypatch.setattr(retrain, "make_alert", lambda *_a, **_k: alerts.append)
    patched_main(
        lambda *_a, **_k: retrain.RetrainResult(
            run_id="r", status="skipped", warnings=["training data not ready"]
        )
    )
    assert retrain.main([]) == 0
    assert alerts == ["training data not ready"]

    patched_main(lambda *_a, **_k: retrain.RetrainResult(run_id="r", failures=["x"]))
    assert retrain.main([]) == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    patched_main(down)
    assert retrain.main([]) == 1

    def bad_config(*_a, **_k):
        raise retrain.ConfigError("nope")

    patched_main(bad_config)
    assert retrain.main([]) == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    patched_main(crash)
    assert retrain.main([]) == 1

    monkeypatch.setattr(retrain, "get_settings", lambda: Settings(auto_promote="maybe"))
    assert retrain.main([]) == 2
