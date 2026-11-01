"""KAN-76: the one-parameter smoother as shadow challenger (registry, predict, score, retrain)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pm25_forecast import predict, score
from pm25_forecast.features import SMOOTHER_FEATURE_NAMES, mean_24h
from pm25_forecast.models import smoother as smoother_build
from pm25_forecast.models.baselines import PERSISTENCE_MODEL_VERSION
from pm25_forecast.models.serving import (
    ModelLoadError,
    SmootherModel,
    load_registered_model,
    load_serving_model,
)
from pm25_forecast.settings import Settings
from pm25_forecast.storage import registry
from test_model_registry import InMemoryRegistry
from test_predict import ISSUE, FakeDb, do_run, load_inputs
from test_predict import no_network as _no_network  # noqa: F401  (autouse: no sockets)

FIXTURES = Path(__file__).parent / "fixtures"
VERSION = smoother_build.SMOOTHER_VERSION
H = timedelta(hours=1)


def registered(artifact=None, manifest=None):
    artifact = artifact or smoother_build.smoother_artifact()
    manifest = manifest or smoother_build.smoother_manifest(artifact)
    reg = InMemoryRegistry()
    registry.write_model(reg, manifest["version"], artifact, manifest)
    return reg


# --- shared feature -----------------------------------------------------------------------------


def test_mean_24h_uses_only_the_o24_to_o1_window_of_available_readings():
    data = json.loads((FIXTURES / "smoother_inputs.json").read_text())
    now = datetime.fromisoformat(data["prediction_time"])
    rows = [(o["ts"], o["pm25"], o["ingested_at"]) for o in data["observations"]]
    rows = [(datetime.fromisoformat(t), v, datetime.fromisoformat(a)) for t, v, a in rows]
    assert mean_24h(rows, now) == data["expected_mean_24h"]


def test_mean_24h_is_none_without_readings_and_rejects_naive_time():
    now = datetime(2026, 10, 28, 12, tzinfo=UTC)
    assert mean_24h([], now) is None
    assert mean_24h([(now - 30 * H, 5.0)], now) is None  # too old
    with pytest.raises(ValueError, match="timezone-aware"):
        mean_24h([], datetime(2026, 10, 28, 12))  # noqa: DTZ001


# --- model --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta, expected", [(0.0, 40.0), (0.5, 35.0), (1.0, 30.0)]
)  # latest 40, mean24 30
def test_smoother_formula(beta, expected):
    model = SmootherModel("m", beta, (24,))
    assert model.predict(24, {"pm25_latest": 40.0, "pm25_mean_24h": 30.0}) == expected


def test_smoother_gaps_and_clamp():
    model = SmootherModel("m", 0.5, (24,))
    assert model.predict(24, {"pm25_latest": None, "pm25_mean_24h": 30.0}) is None
    assert model.predict(24, {"pm25_latest": 40.0, "pm25_mean_24h": None}) == 40.0  # persistence
    assert SmootherModel("m", 5.0, (24,)).predict(24, {"pm25_latest": 1, "pm25_mean_24h": 0}) == 0.0
    with pytest.raises(ModelLoadError, match="did not produce"):
        model.predict(24, {"pm25_latest": 1.0})


def test_build_beta_is_the_kan65_choice_and_in_the_manifest():
    assert smoother_build.SMOOTHER_BETA == 0.5
    artifact = smoother_build.smoother_artifact()
    manifest = smoother_build.smoother_manifest(artifact)
    assert json.loads(artifact)["smoother"] == {"beta": 0.5, "horizons": [24]}
    assert manifest["feature_list"] == list(SMOOTHER_FEATURE_NAMES)
    assert "KAN-65" in manifest["beta_source"] and manifest["assumed_reporting_lag_hours"] == 1


# --- registry and loader ------------------------------------------------------------------------


def test_registered_smoother_loads_through_the_registry():
    model = load_registered_model(registered(), VERSION)
    assert isinstance(model, SmootherModel)
    assert (model.beta, model.horizons, model.feature_list) == (0.5, (24,), SMOOTHER_FEATURE_NAMES)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"format": 1, "smoother": {"beta": 1.5, "horizons": [24]}}, "beta"),
        ({"format": 1, "smoother": {"beta": "x", "horizons": [24]}}, "beta"),
    ],
)
def test_registry_rejects_a_bad_smoother_beta(payload, message):
    artifact = json.dumps(payload).encode()
    manifest = smoother_build.smoother_manifest(artifact)
    with pytest.raises(ValueError, match=message):
        registry.write_model(InMemoryRegistry(), VERSION, artifact, manifest)


def test_registry_rejects_smoother_with_other_features():
    artifact = smoother_build.smoother_artifact()
    manifest = {**smoother_build.smoother_manifest(artifact), "feature_list": ["pm25_latest"]}
    with pytest.raises(ValueError, match="smoother inputs"):
        registry.write_model(InMemoryRegistry(), VERSION, artifact, manifest)


@pytest.mark.parametrize(
    "spec",
    [{"beta": 0.5}, {"beta": 0.5, "horizons": []}, {"beta": 0.5, "horizons": [48]}],
)
def test_loader_rejects_bad_smoother_horizons(spec):
    artifact = json.dumps({"format": 1, "smoother": spec}).encode()
    with pytest.raises(ModelLoadError):
        load_registered_model(registered(artifact), VERSION)


def test_lightgbm_loader_refuses_a_smoother_bundle():
    artifact = smoother_build.smoother_artifact()
    with pytest.raises(ModelLoadError):
        load_serving_model(VERSION, artifact, smoother_build.smoother_manifest(artifact))


# --- predict (shadow) and score -----------------------------------------------------------------


def test_predict_writes_smoother_rows_as_challenger_next_to_persistence(monkeypatch):
    reg = registered()
    monkeypatch.setattr(
        predict, "load_registered_model", lambda _con, v: load_registered_model(reg, v)
    )
    fake = FakeDb()
    load_inputs(fake)
    result = do_run(fake, challenger_model_version=VERSION)
    assert not result.failures
    rows = fake.predictions()
    challenger = [r for r in rows if r[3] == VERSION]
    # 59t: latest 40 (12:00), mean24 = 38 (only 11:00 is in o-24..o-1) -> 40 + 0.5 * (38 - 40)
    assert [(r[0], r[2], r[5]) for r in challenger] == [("59t", (ISSUE + 24 * H).isoformat(), 39.0)]
    assert result.per_model[VERSION] == 1
    assert any(r[3] == PERSISTENCE_MODEL_VERSION and r[5] == 40.0 for r in rows)  # served as before
    assert any("61t: no inputs for " + VERSION in m for m in result.skipped)


def test_score_picks_up_the_smoother_challenger():
    from test_score import add_obs, add_pred

    fake = FakeDb()
    issue = datetime(2026, 10, 29, 12, tzinfo=UTC)
    add_pred(fake, "59t", issue, 24, PERSISTENCE_MODEL_VERSION, 40.0)
    add_pred(fake, "59t", issue, 24, VERSION, 35.0)
    add_obs(fake, "59t", issue + 24 * H, 30.0)
    settings = Settings(
        _env_file=None, database_url=predict_url(), challenger_model_version=VERSION
    )
    score.run(settings, connect=fake, clock=lambda: datetime(2026, 10, 30, 12, 20, tzinfo=UTC))
    got = fake.con.execute("SELECT model_version, abs_error FROM scores ORDER BY 1").fetchall()
    assert got == [(VERSION, 5.0), (PERSISTENCE_MODEL_VERSION, 10.0)]


def predict_url():
    from test_predict import DB_URL

    return DB_URL


def test_retrain_reports_a_smoother_challenger_as_skipped_not_crashing():
    from test_retrain import NOW, Registry, do_run
    from test_retrain import settings as retrain_settings

    reg = Registry()
    artifact = smoother_build.smoother_artifact()
    registry.write_model(reg, VERSION, artifact, smoother_build.smoother_manifest(artifact))
    result = do_run(reg, [], now=NOW, settings=retrain_settings(challenger_model_version=VERSION))
    assert result.status == "ok"
    assert result.challenger_comparison == {}
    assert "scored by Score" in result.roles["challenger_error"]
