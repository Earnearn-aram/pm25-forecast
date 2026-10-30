"""KAN-24: model_events log, rollback, registry prune, drift-triggered retrain. Offline."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast import registry_cli, retrain
from pm25_forecast.retrain import Decision
from pm25_forecast.settings import Settings
from pm25_forecast.storage import registry
from test_model_registry import InMemoryRegistry
from test_predict import FakeDb
from test_retrain import NOW as RETRAIN_NOW
from test_retrain import Registry, do_run, settings
from test_serving import KAN15_FEATURES, manifest, single_bundle

# --- registry with a log ----------------------------------------------------------------------


def reg_with(*names, bad=()):
    """In-memory registry holding servable models ``names`` and unservable ``bad`` ones."""
    reg = InMemoryRegistry()
    art = single_bundle()
    for name in names:
        man = manifest(name, art, KAN15_FEATURES)
        reg.rows[name] = (registry.artifact_sha256(art), art, json.dumps(man))
    bad_features = ["pm25_latest", "humidity"]
    bad_art = single_bundle(bad_features)
    for name in bad:
        man = manifest(name, bad_art, bad_features)
        reg.rows[name] = (registry.artifact_sha256(bad_art), bad_art, json.dumps(man))
    return reg


def cli(reg, *argv):
    lines = []
    code = registry_cli.run(
        list(argv), database_url="db", connect=lambda _url: reg, out=lines.append
    )
    return code, "\n".join(lines)


def test_promote_writes_one_event_per_change_with_the_previous_champion():
    reg = reg_with("a", "b")
    assert registry.promote_model(reg, "a", actor="cli", reason="first") is None
    assert registry.promote_model(reg, "b", actor="auto", metrics={"n_pairs": 120}) == "a"
    first, second = reg.events
    assert (first["action"], first["model_version"]) == ("promote", "a")
    assert first["previous_version"] is None
    assert (second["action"], second["model_version"]) == ("promote", "b")
    assert second["previous_version"] == "a"
    assert first["actor"] == "cli" and first["reason"] == "first"
    assert second["actor"] == "auto" and json.loads(second["metrics"]) == {"n_pairs": 120}
    assert second["at"].tzinfo is not None


def test_event_metrics_with_nan_become_valid_json_null():
    reg = reg_with("a")
    registry.promote_model(reg, "a", metrics={"mae": float("nan"), "nested": [1.0, float("inf")]})
    assert json.loads(reg.events[0]["metrics"]) == {"mae": None, "nested": [1.0, None]}


def test_a_failed_promote_writes_no_event():
    reg = reg_with("a")
    with pytest.raises(LookupError):
        registry.promote_model(reg, "missing")
    assert reg.events == [] and reg.champion is None


def test_record_event_rejects_unknown_action_actor_and_naive_time():
    reg = reg_with("a")
    with pytest.raises(ValueError, match="action"):
        registry.record_event(reg, "delete", "a")
    with pytest.raises(ValueError, match="actor"):
        registry.record_event(reg, "keep", "a", actor="robot")
    with pytest.raises(ValueError, match="timezone-aware"):
        registry.record_event(reg, "keep", "a", now=datetime(2026, 1, 1))  # noqa: DTZ001
    assert reg.events == []


def test_model_events_sql_round_trips_in_the_real_schema():
    fake = FakeDb()
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    fake.con.execute("INSERT INTO model_champion (slot, model_version) VALUES ('production', 'b')")
    registry.record_event(fake, "promote", "a", actor="cli", now=t0)
    day = timedelta(days=1)
    registry.record_event(fake, "promote", "b", previous_version="a", actor="auto", now=t0 + day)
    registry.record_event(
        fake, "recommend", "c", previous_version="b", actor="retrain", now=t0 + 2 * day
    )
    assert registry.previous_champion(fake) == "a"  # a recommend never counts as a change
    with pytest.raises(Exception, match="CHECK|constraint"):
        fake.con.execute(
            "INSERT INTO model_events (at, action, model_version, actor) "
            "VALUES ('x', 'delete', 'z', 'cli')"
        )


# --- rollback ---------------------------------------------------------------------------------


def test_rollback_moves_back_to_the_previous_champion_and_logs_it():
    reg = reg_with("a", "b")
    cli(reg, "promote", "a", "--confirm")
    cli(reg, "promote", "b", "--confirm")
    assert reg.champion == "b"
    code, text = cli(reg, "rollback", "--confirm")
    assert code == 0 and reg.champion == "a"
    assert text == "production champion: b -> a (rollback)"
    last = reg.events[-1]
    assert (last["action"], last["model_version"]) == ("rollback", "a")
    assert last["previous_version"] == "b" and last["actor"] == "cli"
    assert len(reg.events) == 3  # one row per champion change


def test_rollback_needs_confirm():
    reg = reg_with("a", "b")
    cli(reg, "promote", "a", "--confirm")
    cli(reg, "promote", "b", "--confirm")
    code, text = cli(reg, "rollback")
    assert code == 1 and "--confirm" in text and reg.champion == "b"


def test_rollback_refuses_when_no_previous_champion_is_recorded():
    reg = reg_with("a")
    code, text = cli(reg, "rollback", "--confirm")
    assert code == 1 and "nothing to roll back" in text  # no champion at all
    cli(reg, "promote", "a", "--confirm")  # the first ever promote has no previous champion
    code, text = cli(reg, "rollback", "--confirm")
    assert code == 1 and "no previous champion" in text and reg.champion == "a"


def test_rollback_refuses_an_unservable_previous_champion():
    reg = reg_with("good", bad=("bad",))
    registry.promote_model(reg, "bad")  # promote_model alone does not check servability
    registry.promote_model(reg, "good")
    code, text = cli(reg, "rollback", "--confirm")
    assert code == 1 and "shared feature builder" in text
    assert reg.champion == "good" and len(reg.events) == 2


def test_rollback_refuses_when_the_log_does_not_match_the_live_pointer():
    reg = reg_with("a", "b")
    registry.promote_model(reg, "a")
    registry.promote_model(reg, "b")
    reg.champion = "a"  # pointer moved outside the log
    code, text = cli(reg, "rollback", "--confirm")
    assert code == 1 and "not the current champion" in text and reg.champion == "a"


def test_manual_promote_is_logged_with_actor_cli():
    reg = reg_with("a")
    assert cli(reg, "promote", "a", "--confirm") == (0, "production champion: none -> a")
    assert [(e["action"], e["actor"], e["reason"]) for e in reg.events] == [
        ("promote", "cli", "manual promote")
    ]


# --- prune ------------------------------------------------------------------------------------

NOW = datetime.now(UTC)


def add_models(fake, names):
    """Models created one day apart, ``names[0]`` the oldest."""
    for i, name in enumerate(names):
        created = (NOW - timedelta(days=len(names) - i)).isoformat()
        fake.con.execute(
            "INSERT INTO models (model_version, sha256, artifact, manifest, created_at) "
            "VALUES (?, 'x', x'00', '{}', ?)",
            (name, created),
        )
    fake.con.commit()


def add_prediction(fake, version, age_days):
    issue = (NOW - timedelta(days=age_days)).isoformat()
    fake.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        " pm25_pred) VALUES ('59t', ?, ?, ?, 'r', 1.0)",
        (issue, issue, version),
    )
    fake.con.commit()


def prune_db():
    fake = FakeDb()
    add_models(fake, [f"m{i}" for i in range(1, 8)])  # m7 newest
    fake.con.execute("INSERT INTO model_champion (slot, model_version) VALUES ('production', 'm1')")
    add_prediction(fake, "m3", age_days=10)  # inside the 30 day window: protected
    add_prediction(fake, "m4", age_days=60)  # outside it: not protected
    fake.con.execute(
        "INSERT INTO drift_reference (model_version, feature, edges, shares, n_ref, n_origins,"
        " usable) VALUES ('m5', 'pm25_latest', '[]', '[]', 1, 1, 0)"
    )
    fake.con.commit()
    return fake


def models_left(fake):
    return sorted(r[0] for r in fake.con.execute("SELECT model_version FROM models"))


def test_plan_keeps_champion_challenger_recent_predictions_and_the_newest():
    plan = registry.plan_prune(prune_db(), 2, challenger="m2", now=NOW)
    assert set(plan.keep) == {"m1", "m2", "m3", "m6", "m7"}
    assert plan.keep["m1"] == "production champion" and plan.keep["m2"] == "challenger"
    assert sorted(plan.delete) == ["m4", "m5"]


def test_plan_with_keep_zero_still_keeps_the_protected_models():
    plan = registry.plan_prune(prune_db(), 0, challenger=None, now=NOW)
    assert set(plan.keep) == {"m1", "m3"}
    assert sorted(plan.delete) == ["m2", "m4", "m5", "m6", "m7"]
    with pytest.raises(ValueError, match="negative"):
        registry.plan_prune(prune_db(), -1)


def test_prune_cli_is_a_dry_run_by_default(monkeypatch):
    monkeypatch.setattr(registry_cli, "get_settings", lambda: Settings(_env_file=None))
    fake = prune_db()
    code, text = cli(fake, "prune", "--keep", "2")
    assert code == 0
    # no challenger configured here, so m2 is also free to go
    assert "delete m4" in text and "delete m5" in text and "delete m2" in text
    assert "dry run: 3 model(s)" in text
    assert "keep m1: production champion" in text
    assert len(models_left(fake)) == 7


def test_prune_cli_confirm_deletes_only_the_planned_models(monkeypatch):
    with_challenger = Settings(_env_file=None, challenger_model_version="m2")
    monkeypatch.setattr(registry_cli, "get_settings", lambda: with_challenger)
    fake = prune_db()
    code, text = cli(fake, "prune", "--keep", "2", "--confirm")
    assert code == 0 and "deleted 2 model(s)" in text
    assert models_left(fake) == ["m1", "m2", "m3", "m6", "m7"]
    left = fake.con.execute("SELECT count(*) FROM drift_reference").fetchone()[0]
    assert left == 0  # the cached drift reference of a deleted model goes with it


def test_prune_never_deletes_a_referenced_model_even_from_a_stale_plan():
    fake = prune_db()
    stale = registry.PrunePlan(keep={}, delete=["m1", "m3", "m5"])  # champion, recent, free
    deleted = registry.prune_models(fake, stale, now=NOW)
    assert deleted == ["m5"]
    assert "m1" in models_left(fake) and "m3" in models_left(fake)


# --- retrain: events and --if-drift -----------------------------------------------------------


def test_auto_promote_logs_a_promote_event_with_the_comparison():
    reg = Registry()
    result = do_run(reg, [], settings=settings(auto_promote=True))
    (event,) = reg.events
    assert (event["action"], event["actor"], event["model_version"]) == (
        "promote",
        "auto",
        result.version,
    )
    assert event["previous_version"] is None
    assert json.loads(event["metrics"])["comparison"]["+24h"]["n_pairs"] >= 100


def test_recommend_only_run_logs_a_recommend_event_and_leaves_the_pointer_alone():
    reg = Registry()
    result = do_run(reg, [])
    (event,) = reg.events
    assert event["action"] == "recommend" and event["actor"] == "retrain"
    assert event["model_version"] == result.version and reg.champion is None
    assert json.loads(event["metrics"])["candidate"] == result.version


def test_keep_decision_is_logged_with_its_reason(monkeypatch):
    keep = Decision("keep", "too small", {"n": 1})
    monkeypatch.setattr(retrain, "decide", lambda *_a, **_k: keep)
    reg = Registry()
    result = do_run(reg, [])
    (event,) = reg.events
    assert (event["action"], event["reason"]) == ("keep", "too small")
    assert event["model_version"] == result.version and reg.champion is None


def boom_export(_path, _url):
    raise AssertionError("an --if-drift run without drift must not export or train")


def test_if_drift_without_an_alert_exits_clean_without_training(monkeypatch):
    calls = []

    def no_alert(_con, model_version=None, *, now=None):
        calls.append(model_version)
        return False

    monkeypatch.setattr(retrain, "latest_drift_alert", no_alert)
    reg = Registry()
    result = do_run(
        reg,
        [],
        export=boom_export,
        if_drift=True,
        settings=settings(challenger_model_version="chal"),
    )
    assert result.status == "skipped" and not result.failures
    assert calls == ["chal"] and reg.rows == {} and reg.events == []


def test_if_drift_with_an_alert_runs_the_normal_recommend_only_retrain(monkeypatch):
    seen = []

    def alert(_con, model_version=None, *, now=None):
        seen.append((model_version, now))
        return True

    monkeypatch.setattr(retrain, "latest_drift_alert", alert)
    reg = Registry()
    # "chal" is not registered: it is skipped in the comparison, the run goes on
    result = do_run(reg, [], if_drift=True, settings=settings(challenger_model_version="chal"))
    assert seen == [("chal", RETRAIN_NOW)]
    assert result.status == "ok" and result.version in reg.rows
    assert result.recommended and not result.promoted and reg.champion is None
    assert [e["action"] for e in reg.events] == ["recommend"]


def test_if_drift_checks_the_champion_when_there_is_no_challenger(monkeypatch):
    seen = []

    def alert(_con, model_version=None, *, now=None):
        seen.append(model_version)
        return False

    monkeypatch.setattr(retrain, "latest_drift_alert", alert)
    reg = Registry()
    reg.champion = "champ"
    result = do_run(reg, [], export=boom_export, if_drift=True)
    assert result.status == "skipped" and seen == ["champ"]


def test_if_drift_with_no_challenger_and_no_champion_does_nothing(monkeypatch):
    monkeypatch.setattr(retrain, "latest_drift_alert", lambda *_a, **_k: pytest.fail("no model"))
    result = do_run(Registry(), [], export=boom_export, if_drift=True)
    assert result.status == "skipped"


def test_if_drift_flag_reaches_run_from_the_command_line(monkeypatch):
    seen = {}
    monkeypatch.setattr(retrain, "get_settings", lambda: settings())

    def fake_run(*_a, **kw):
        seen.update(kw)
        return retrain.RetrainResult(run_id="r", status="skipped")

    monkeypatch.setattr(retrain, "run", fake_run)
    assert retrain.main(["--if-drift"]) == 0
    assert seen["if_drift"] is True
    assert retrain.main([]) == 0 and seen["if_drift"] is False
