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
    assert "at" not in second  # the event time is the database clock (column default)


def test_event_metrics_with_nan_become_valid_json_null():
    reg = reg_with("a")
    registry.promote_model(reg, "a", metrics={"mae": float("nan"), "nested": [1.0, float("inf")]})
    assert json.loads(reg.events[0]["metrics"]) == {"mae": None, "nested": [1.0, None]}


def test_a_failed_promote_writes_no_event():
    reg = reg_with("a")
    with pytest.raises(LookupError):
        registry.promote_model(reg, "missing")
    assert reg.events == [] and reg.champion is None


def test_record_event_rejects_unknown_action_and_actor():
    reg = reg_with("a")
    with pytest.raises(ValueError, match="action"):
        registry.record_event(reg, "delete", "a")
    with pytest.raises(ValueError, match="actor"):
        registry.record_event(reg, "keep", "a", actor="robot")
    assert reg.events == []


class LockFreeDb(FakeDb):
    """FakeDb for the real model_events SQL: SQLite has no `FOR UPDATE`, so it is dropped."""

    def execute(self, statement, params=()):
        return super().execute(statement.replace(" FOR UPDATE", ""), params)


def test_lock_statement_is_the_champion_select_for_update():
    assert registry.SELECT_CHAMPION_LOCK.strip().endswith("FOR UPDATE")
    assert registry.SELECT_CHAMPION_LOCK.startswith(registry.SELECT_CHAMPION.rstrip("\n"))


def test_promote_and_rollback_lock_the_pointer_before_reading_it():
    reg = reg_with("a", "b")
    seen = []
    plain = reg.execute

    def spy(statement, params):
        seen.append(statement)
        return plain(statement, params)

    reg.execute = spy
    registry.promote_model(reg, "a")
    registry.promote_model(reg, "b")
    assert registry.previous_champion(reg) == "a"
    first_champion_reads = [s for s in seen if "FROM model_champion" in s and "SELECT" in s]
    assert first_champion_reads[0] == registry.SELECT_CHAMPION_LOCK  # promote: before the upsert
    assert seen.index(registry.SELECT_CHAMPION_LOCK) < seen.index(registry.UPSERT_CHAMPION)
    assert seen[-2] == registry.SELECT_CHAMPION_LOCK  # previous_champion: lock, then the log
    assert seen[-1] == registry.SELECT_LAST_CHANGE


def test_model_events_sql_round_trips_and_orders_by_id_in_the_real_schema():
    fake = LockFreeDb()
    fake.con.execute("INSERT INTO model_champion (slot, model_version) VALUES ('production', 'b')")
    # All three rows get the same database timestamp (CURRENT_TIMESTAMP, one second resolution):
    # only `id` can order them.
    registry.record_event(fake, "promote", "a", actor="cli")
    registry.record_event(fake, "promote", "b", previous_version="a", actor="auto")
    registry.record_event(fake, "recommend", "c", previous_version="b", actor="retrain")
    assert registry.previous_champion(fake) == "a"  # a recommend never counts as a change
    ids = [r[0] for r in fake.con.execute("SELECT id FROM model_events ORDER BY id")]
    assert ids == [1, 2, 3]
    assert fake.con.execute("SELECT count(*) FROM model_events WHERE at IS NULL").fetchone()[0] == 0
    with pytest.raises(Exception, match=r"CHECK|constraint"):
        fake.con.execute(
            "INSERT INTO model_events (action, model_version, actor) VALUES ('delete', 'z', 'cli')"
        )


def test_last_retrain_event_covers_retrain_and_auto_but_not_cli():
    fake = LockFreeDb()
    assert registry.last_retrain_event_at(fake) is None
    registry.record_event(fake, "promote", "a", actor="cli")
    assert registry.last_retrain_event_at(fake) is None
    registry.record_event(fake, "keep", "b", actor="retrain")
    found = registry.last_retrain_event_at(fake)
    assert found is not None and found.tzinfo is not None
    assert abs(datetime.now(UTC) - found) < timedelta(minutes=5)


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


def test_prune_keeps_the_rollback_target_and_never_deletes_it():
    fake = prune_db()
    registry.record_event(fake, "promote", "m1", previous_version="m5", actor="cli")
    plan = registry.plan_prune(fake, 2, challenger="m2", now=NOW)
    assert plan.keep["m5"] == "rollback target (previous champion)"
    assert plan.delete == ["m4"]
    stale = registry.PrunePlan(keep={}, delete=["m5", "m4"])  # a plan made without the target
    assert registry.prune_models(fake, stale, now=NOW) == ["m4"]
    assert "m5" in models_left(fake)


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


def patch_drift(monkeypatch, alerting, computed=RETRAIN_NOW - timedelta(hours=1)):
    """Fake the drift module: models in ``alerting`` have an alert; the run was at ``computed``."""
    seen = []

    def alert(_con, model_version=None, *, now=None):
        seen.append((model_version, now))
        return model_version in alerting

    monkeypatch.setattr(retrain, "latest_drift_alert", alert)
    monkeypatch.setattr(retrain, "latest_drift_run", lambda _con, model_version: computed)
    return seen


def test_if_drift_without_an_alert_exits_clean_without_training(monkeypatch):
    seen = patch_drift(monkeypatch, alerting=())
    reg = Registry()
    result = do_run(
        reg,
        [],
        export=boom_export,
        if_drift=True,
        settings=settings(challenger_model_version="chal"),
    )
    assert result.status == "skipped" and not result.failures
    assert seen == [("chal", RETRAIN_NOW)] and reg.rows == {} and reg.events == []


def test_if_drift_with_an_alert_runs_the_normal_recommend_only_retrain(monkeypatch):
    patch_drift(monkeypatch, alerting=("chal",))
    reg = Registry()
    # "chal" is not registered: it is skipped in the comparison, the run goes on
    result = do_run(reg, [], if_drift=True, settings=settings(challenger_model_version="chal"))
    assert result.status == "ok" and result.version in reg.rows
    assert result.recommended and not result.promoted and reg.champion is None
    assert [e["action"] for e in reg.events] == ["recommend"]


def test_drift_gate_checks_both_the_challenger_and_the_champion(monkeypatch):
    reg = Registry()
    reg.champion = "champ"
    both = settings(challenger_model_version="chal")
    seen = patch_drift(monkeypatch, alerting=("champ",))
    assert retrain.drift_gate(reg, both, RETRAIN_NOW) == (["champ"], None)
    assert [m for m, _ in seen] == ["chal", "champ"]  # the challenger's silence is not enough
    patch_drift(monkeypatch, alerting=("chal",))
    assert retrain.drift_gate(reg, both, RETRAIN_NOW) == (["chal"], None)
    patch_drift(monkeypatch, alerting=("chal", "champ"))
    assert retrain.drift_gate(reg, both, RETRAIN_NOW) == (["chal", "champ"], None)
    patch_drift(monkeypatch, alerting=())
    assert retrain.drift_gate(reg, both, RETRAIN_NOW) == ([], "no drift alert")


def test_drift_gate_checks_the_champion_alone_when_there_is_no_challenger(monkeypatch):
    reg = Registry()
    reg.champion = "champ"
    seen = patch_drift(monkeypatch, alerting=("champ",))
    assert retrain.drift_gate(reg, settings(), RETRAIN_NOW) == (["champ"], None)
    assert [m for m, _ in seen] == ["champ"]


def test_if_drift_with_no_challenger_and_no_champion_does_nothing(monkeypatch):
    monkeypatch.setattr(retrain, "latest_drift_alert", lambda *_a, **_k: pytest.fail("no model"))
    result = do_run(Registry(), [], export=boom_export, if_drift=True)
    assert result.status == "skipped"


def test_cooldown_skips_when_a_retrain_was_logged_after_the_drift_run(monkeypatch):
    patch_drift(monkeypatch, alerting=("chal",), computed=RETRAIN_NOW - timedelta(hours=1))
    reg = Registry()
    reg.last_retrain_at = RETRAIN_NOW - timedelta(minutes=5)  # a retrain ran after the drift run
    chal = settings(challenger_model_version="chal")
    alerted, reason = retrain.drift_gate(reg, chal, RETRAIN_NOW)
    assert alerted == ["chal"] and reason.startswith("cooldown")
    result = do_run(reg, [], export=boom_export, if_drift=True, settings=chal)
    assert result.status == "skipped" and reg.rows == {}


def test_a_retrain_before_the_drift_run_does_not_block_it(monkeypatch):
    patch_drift(monkeypatch, alerting=("chal",), computed=RETRAIN_NOW - timedelta(hours=1))
    reg = Registry()
    reg.last_retrain_at = RETRAIN_NOW - timedelta(days=3)  # last week's retrain
    gate = retrain.drift_gate(reg, settings(challenger_model_version="chal"), RETRAIN_NOW)
    assert gate == (["chal"], None)


def test_one_drift_alert_triggers_one_retrain_even_if_the_job_is_rerun(monkeypatch):
    patch_drift(monkeypatch, alerting=("chal",))
    reg = Registry()
    chal = settings(challenger_model_version="chal")
    first = do_run(reg, [], if_drift=True, settings=chal)
    assert first.status == "ok" and len(reg.events) == 1  # the run logged a recommend event
    again = do_run(reg, [], export=boom_export, if_drift=True, settings=chal)
    assert again.status == "skipped" and len(reg.events) == 1


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
