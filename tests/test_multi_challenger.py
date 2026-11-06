"""KAN-104: several shadow challengers side by side, ridge-style hourly horizons, 80 h history.

Offline: SQLite built from schema.sql, models are small stand-ins (or a hand-built RidgeModel), no
network. The daily live comparison and the promotion rule are KAN-110, not tested here.
"""

import subprocess
import sys
from datetime import timedelta

import pytest
from pydantic import ValidationError

from pm25_forecast import predict, score
from pm25_forecast.models.baselines import PERSISTENCE_MODEL_VERSION
from pm25_forecast.models.ridge_bundle import RidgeHorizonSpec, RidgeSpec
from pm25_forecast.models.serving import RIDGE_LOOKBACK, ModelLoadError, RidgeModel
from pm25_forecast.settings import MAX_CHALLENGERS, Settings, parse_challenger_versions
from pm25_forecast.storage import registry
from test_predict import (
    DB_URL,
    ISSUE,
    NOW,
    FakeDb,
    H,
    do_run,
    load_inputs,
    no_network,  # noqa: F401  (autouse fixture: no sockets in tests)
    settings,
)


class Stub:
    """A servable model stand-in: fixed value, own horizon list, optional failure."""

    def __init__(self, version, horizons, value, fail=False):
        self.model_version = version
        self.horizons = horizons
        self.value = value
        self.fail = fail

    def predict(self, horizon, features):
        if self.fail:
            raise ValueError("boom")
        if features["pm25_latest"] is None:
            return None  # like the real models: nothing to predict from
        return self.value + horizon / 100


@pytest.fixture
def db():
    fake = FakeDb()
    load_inputs(fake)
    return fake


def serve(monkeypatch, **models):
    """Registry stand-in: a model object per version, an exception instance is raised."""

    def load(_con, version):
        found = models[version.replace("-", "_")]
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(predict, "load_registered_model", load)


def rows_of(fake, version):
    return {r[2]: r[5] for r in fake.predictions() if r[3] == version and r[0] == "59t"}


# --- configuration -----------------------------------------------------------------------------


def test_unset_and_blank_mean_no_challenger():
    assert Settings(_env_file=None).challenger_model_versions == ()
    blank = Settings(_env_file=None, challenger_model_version="   ")
    assert blank.challenger_model_version is None and blank.challenger_model_versions == ()


def test_single_value_is_unchanged():
    one = Settings(_env_file=None, challenger_model_version="  kan15-test ")
    assert one.challenger_model_version == "kan15-test"  # trimmed exactly as before
    assert one.challenger_model_versions == ("kan15-test",)


def test_list_is_trimmed_and_ordered():
    s = Settings(_env_file=None, challenger_model_version=" a , b,c ")
    assert s.challenger_model_versions == ("a", "b", "c")
    assert s.challenger_model_version == "a,b,c"


@pytest.mark.parametrize(
    "raw",
    ["a,b,c,d", "a,a", "a, a ", "a,,b", "a,", ",a", ","],
    ids=[
        "cap",
        "duplicate",
        "duplicate-spaced",
        "empty-middle",
        "trailing",
        "leading",
        "only-comma",
    ],
)
def test_invalid_lists_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_challenger_versions(raw)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, challenger_model_version=raw)
    assert MAX_CHALLENGERS == 3


def test_every_listed_version_must_differ_from_the_baselines(db):
    with pytest.raises(predict.ConfigError, match="must differ"):
        do_run(db, challenger_model_version=f"ok-model,{PERSISTENCE_MODEL_VERSION}")
    with pytest.raises(predict.ConfigError, match="must differ"):
        do_run(db, challenger_model_version="ok-model,cams")


def test_a_bad_list_is_a_config_error_exit_2_not_a_crash(monkeypatch):
    monkeypatch.setenv("CHALLENGER_MODEL_VERSION", "a,a")
    monkeypatch.setattr(predict, "get_settings", lambda: Settings(_env_file=None))
    assert predict.main() == predict.EXIT_CONFIG
    monkeypatch.setattr(score, "get_settings", lambda: Settings(_env_file=None))
    assert score.main([]) == score.EXIT_CONFIG


# --- predict -----------------------------------------------------------------------------------


def test_two_challengers_are_stored_under_their_own_versions_and_horizons(db, monkeypatch):
    serve(
        monkeypatch,
        first=Stub("first", (6, 24), 10.0),
        second=Stub("second", (1, 2, 24), 20.0),
    )
    result = do_run(db, challenger_model_version="first, second")
    assert result.failures == []
    t = lambda n: (ISSUE + n * H).isoformat()  # noqa: E731
    assert rows_of(db, "first") == {t(6): 10.06, t(24): 10.24}
    assert rows_of(db, "second") == {t(1): 20.01, t(2): 20.02, t(24): 20.24}
    assert len(rows_of(db, PERSISTENCE_MODEL_VERSION)) == 24
    assert result.per_model == {PERSISTENCE_MODEL_VERSION: 24, "first": 2, "second": 3}


def test_one_failing_challenger_fails_the_run_but_not_the_other(db, monkeypatch):
    for broken in (
        ModelLoadError("bad bundle"),  # cannot be loaded
        Stub("first", (24,), 10.0, fail=True),  # loads, then fails while predicting
    ):
        fresh = FakeDb()
        load_inputs(fresh)
        serve(monkeypatch, first=broken, second=Stub("second", (6, 24), 20.0))
        result = do_run(fresh, challenger_model_version="first,second")
        assert len(result.failures) == 1 and "first" in result.failures[0]
        assert rows_of(fresh, "first") == {}
        assert len(rows_of(fresh, "second")) == 2  # the healthy challenger is still written
        assert len(rows_of(fresh, PERSISTENCE_MODEL_VERSION)) == 24  # and so are the baselines
        assert result.per_model.get("first", 0) == 0 and result.per_model["second"] == 2


def test_failure_in_the_second_challenger_keeps_the_first(db, monkeypatch):
    serve(monkeypatch, first=Stub("first", (24,), 10.0), second=ModelLoadError("nope"))
    result = do_run(db, challenger_model_version="first,second")
    assert [f.split(" not served")[0] for f in result.failures] == ["challenger second"]
    assert len(rows_of(db, "first")) == 1


def test_a_failed_challenger_still_fails_the_command_loudly(db, monkeypatch):
    serve(monkeypatch, first=ModelLoadError("nope"), second=Stub("second", (24,), 20.0))
    alerts = []
    real_run = predict.run
    cfg = settings(challenger_model_version="first,second")
    monkeypatch.setattr(predict, "safe_alert", lambda _alert, message: alerts.append(message))
    monkeypatch.setattr(predict, "get_settings", lambda: cfg)
    monkeypatch.setattr(predict, "run", lambda s: real_run(s, connect=db, clock=lambda: NOW))
    assert predict.main() == predict.EXIT_FAILED
    assert alerts and "challenger first not served" in alerts[0]
    assert len(rows_of(db, "second")) == 1  # written all the same


def test_single_challenger_behaves_as_before(db, monkeypatch):
    stub = Stub("only", (24,), 33.0)
    serve(monkeypatch, only=stub)
    as_single = do_run(db, challenger_model_version="only")
    single_rows = db.predictions()
    other = FakeDb()
    load_inputs(other)
    as_list = do_run(other, challenger_model_version=" only ")
    assert [r[:4] + r[5:] for r in other.predictions()] == [r[:4] + r[5:] for r in single_rows]
    assert as_single.per_model == as_list.per_model == {PERSISTENCE_MODEL_VERSION: 24, "only": 1}


def test_a_challenger_that_is_also_the_champion_is_written_once(db, monkeypatch):
    serve(monkeypatch, champ=Stub("champ", (24,), 5.0), other=Stub("other", (24,), 6.0))
    db.con.execute("INSERT INTO model_champion(slot, model_version) VALUES ('production','champ')")
    result = do_run(db, challenger_model_version="champ,other")
    assert result.failures == []
    assert len(rows_of(db, "champ")) == 1 and len(rows_of(db, "other")) == 1


def test_missing_inputs_are_summarised_once_per_station_and_model(db, monkeypatch):
    class NoInputs(Stub):
        def predict(self, horizon, features):
            return None

    serve(monkeypatch, quiet=NoInputs("quiet", (6, 12, 24), 0.0))
    result = do_run(db, challenger_model_version="quiet")
    assert result.failures == []
    lines = [m for m in result.skipped if "quiet" in m]
    assert sorted(lines) == [
        "59t: no inputs for quiet +6,12,24h, skipped",
        "61t: no inputs for quiet +6,12,24h, skipped",
    ]
    assert predict._hours_label(list(range(1, 25))) == "+1-24h"
    assert predict._hours_label([1, 2, 24]) == "+1,2,24h"
    assert predict._hours_label([24]) == "+24h"


# --- ridge: 80 h of history, hourly horizons ---------------------------------------------------


def ridge_model(version="ridge-test", horizons=(1, 6, 24)):
    """latest + 1 + 0.1 * mean of the last 72 h: depends on 72 h of history by construction."""
    fit = RidgeHorizonSpec(
        intercept=1.0,
        coef=(0.1, 0.0, 0.0, 0.0),  # one per feature, then one per station
        mean=(0.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0, 1.0),
        strength=1.0,
        n_train=10,
    )
    spec = RidgeSpec(
        feature_set="fs_v2_test",
        features=("pm25_mean_72h",),
        stations=("59t", "61t", "other"),
        missing_policy="skip",
        horizons=dict.fromkeys(horizons, fit),
    )
    return RidgeModel(version, spec)


def add_history(fake, hours=79):
    """59t gets an old plateau (values 100) from -79 h to -25 h, then the fixture's own readings."""
    for k in range(25, hours + 1):
        ts = (ISSUE - k * H).isoformat()
        fake.con.execute(
            "INSERT INTO observations (station_id, ts, pm25, source, ingested_at) "
            "VALUES ('59t', ?, 100.0, 'test', ?)",
            (ts, ts),
        )
    fake.con.commit()


def test_the_ridge_gets_80_hours_of_history_and_hourly_valid_times(db, monkeypatch):
    add_history(db)
    model = ridge_model()
    assert model.lookback == RIDGE_LOOKBACK
    serve(monkeypatch, ridge_test=model)
    result = do_run(db, challenger_model_version="ridge-test")
    assert result.failures == []
    got = rows_of(db, "ridge-test")
    assert sorted(got) == sorted((ISSUE + h * H).isoformat() for h in (1, 6, 24))
    # expected from the same model on the full history; a 24 h read would give a different mean
    full = db.con.execute(
        "SELECT ts, pm25, ingested_at FROM observations WHERE station_id = '59t'"
        " AND pm25 IS NOT NULL AND ts <= ? ORDER BY ts",
        (ISSUE.isoformat(),),
    ).fetchall()
    want = model.forecast(
        [(predict.datetime.fromisoformat(t), p, None) for t, p, _ in full], "59t", ISSUE
    )
    assert got[(ISSUE + 6 * H).isoformat()] == pytest.approx(want[6])
    short = model.forecast(
        [
            (predict.datetime.fromisoformat(t), p, None)
            for t, p, _ in full
            if t >= (ISSUE - 24 * H).isoformat()
        ],
        "59t",
        ISSUE,
    )
    assert short[6] != pytest.approx(want[6])
    assert result.per_model["ridge-test"] == 3
    # 61t (4 h old only) and the station the ridge was not fitted on get no ridge forecast
    assert {r[0] for r in db.predictions() if r[3] == "ridge-test"} == {"59t"}
    assert any("61t: no inputs for ridge-test +1,6,24h" in m for m in result.skipped)


def test_ridge_and_a_booster_style_challenger_side_by_side(db, monkeypatch):
    add_history(db)
    serve(monkeypatch, ridge_test=ridge_model(horizons=(1, 24)), stub=Stub("stub", (6, 24), 7.0))
    result = do_run(db, challenger_model_version="ridge-test,stub")
    assert result.failures == []
    assert len(rows_of(db, "ridge-test")) == 2 and len(rows_of(db, "stub")) == 2


def test_lookback_stays_24_hours_without_a_ridge(db, monkeypatch):
    seen = []
    real = predict.build_station_rows

    def spy(*args, **kwargs):
        seen.append(kwargs["lookback"])
        return real(*args, **kwargs)

    monkeypatch.setattr(predict, "build_station_rows", spy)
    serve(monkeypatch, stub=Stub("stub", (24,), 7.0))
    do_run(db, challenger_model_version="stub")
    assert set(seen) == {timedelta(hours=24)}


# --- score: two challengers, any horizon, side by side with persistence -------------------------


def test_predicted_challengers_are_scored_side_by_side(db, monkeypatch):
    serve(monkeypatch, first=Stub("first", (1, 24), 10.0), second=Stub("second", (6, 24), 20.0))
    cfg = "first,second"
    do_run(db, challenger_model_version=cfg)
    for h in range(1, 25):  # the actuals arrive
        ts = (ISSUE + h * H).isoformat()
        db.con.execute(
            "INSERT INTO observations (station_id, ts, pm25, source) VALUES ('59t', ?, 30.0, 't')",
            (ts,),
        )
    db.con.commit()
    later = ISSUE + timedelta(hours=26)
    result = score.run(
        Settings(_env_file=None, database_url=DB_URL, challenger_model_version=cfg),
        connect=db,
        clock=lambda: later,
    )
    assert result.failures == []
    assert set(result.metrics) == {PERSISTENCE_MODEL_VERSION, "first", "second"}
    assert set(result.metrics["first"]) == {1, 24} and set(result.metrics["second"]) == {6, 24}
    assert set(result.metrics[PERSISTENCE_MODEL_VERSION]) == set(range(1, 25))
    assert result.metrics["first"][1].n_matched == 1  # persistence at +1 h exists to compare with


# --- housekeeping / registry -------------------------------------------------------------------


def test_prune_keeps_every_listed_challenger():
    class Con:
        def execute(self, statement, params=()):
            rows = {
                registry.SELECT_ALL_MODELS: [("m4",), ("m3",), ("m2",), ("m1",)],
                registry.SELECT_CHAMPION: [("m1",)],
            }.get(statement, [])
            return type(
                "Cur",
                (),
                {"fetchall": lambda s: rows, "fetchone": lambda s: rows[0] if rows else None},
            )()

    for challenger, kept in (
        ("m3", {"m1", "m3"}),
        (("m2", "m3"), {"m1", "m2", "m3"}),
        (None, {"m1"}),
    ):
        plan = registry.plan_prune(Con(), 0, challenger=challenger)
        assert set(plan.keep) == kept and set(plan.delete) == {"m1", "m2", "m3", "m4"} - kept


def test_retrain_drift_targets_cover_every_challenger_and_the_champion():
    from pm25_forecast import retrain

    class Con:
        def __init__(self, champion):
            self.champion = champion

        def execute(self, statement, params=()):
            row = None if self.champion is None else (self.champion,)
            return type("Cur", (), {"fetchone": lambda s: row})()

    two = Settings(_env_file=None, challenger_model_version="a, b")
    assert retrain.drift_targets(Con("champ"), two) == ["a", "b", "champ"]
    assert retrain.drift_targets(Con("b"), two) == ["a", "b"]
    assert retrain.drift_targets(Con(None), Settings(_env_file=None)) == []


# --- production modules stay free of experiment-only dependencies ------------------------------


def test_production_modules_import_no_experiments_mlflow_sklearn_or_statsmodels():
    code = (
        "import sys, lightgbm;"  # lightgbm itself imports scikit-learn when installed
        "before = set(sys.modules);"
        "import pm25_forecast.predict, pm25_forecast.score, pm25_forecast.settings,"
        " pm25_forecast.housekeeping, pm25_forecast.models.serving, pm25_forecast.registry_cli;"
        "bad = [m for m in set(sys.modules) - before if m.startswith(('sklearn', 'statsmodels',"
        " 'mlflow', 'pm25_forecast.experiments'))];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
