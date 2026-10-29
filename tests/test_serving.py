"""KAN-29: serving loader, shadow challenger in the predict step, and the manual promote CLI."""

import json
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pytest

from pm25_forecast import predict, registry_cli
from pm25_forecast.models.baselines import CAMS_PASSTHROUGH_MODEL_VERSION, PERSISTENCE_MODEL_VERSION
from pm25_forecast.models.serving import ModelLoadError, load_registered_model, load_serving_model
from pm25_forecast.storage import registry
from test_model_registry import InMemoryRegistry
from test_predict import (
    ISSUE,
    FakeDb,
    do_run,
    load_inputs,
    no_network,  # noqa: F401  (autouse fixture: no sockets in tests)
)

KAN15_FEATURES = ["pm25_latest", "pm25_same_hour_24h"]
KAN32_FEATURES = ["pm25_latest", "pm25_same_hour_24h", "cams_pm25"]


def booster_text(features, offset=0.0):
    x = np.asarray([[float(i + j) for j in range(len(features))] for i in range(30)])
    y = x[:, 0] + offset
    params = {"objective": "regression", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1}
    return lgb.train(params, lgb.Dataset(x, label=y, feature_name=features), 5).model_to_string()


def single_bundle(features=KAN15_FEATURES):
    payload = {"format": 1, "horizon_hours": 24, "booster": booster_text(features)}
    return json.dumps(payload, separators=(",", ":")).encode()


def multi_bundle(features=KAN32_FEATURES):
    text = booster_text(features)
    payload = {"format": 1, "horizons": {str(h): text for h in (6, 12, 24)}}
    return json.dumps(payload, separators=(",", ":")).encode()


def manifest(version, artifact, features):
    return {
        "version": version,
        "metrics": {},
        "training_window": {
            "start": "2024-10-28T00:00:00+00:00",
            "end": "2026-07-01T00:00:00+00:00",
        },
        "feature_list": list(features),
        "data_sha256": "a" * 64,
        "artifact_sha256": registry.artifact_sha256(artifact),
    }


# --- loader ----------------------------------------------------------------------------------


def test_kan15_single_horizon_bundle_loads_with_its_own_feature_subset():
    art = single_bundle()
    model = load_serving_model("kan15", art, manifest("kan15", art, KAN15_FEATURES))
    assert model.horizons == (24,)
    assert model.feature_list == tuple(KAN15_FEATURES)
    pred = model.predict(24, {"pm25_latest": 20.0, "pm25_same_hour_24h": None, "cams_pm25": 9.0})
    assert pred is not None and pred >= 0.0


def test_kan32_multi_horizon_bundle_loads_all_horizons():
    art = multi_bundle()
    model = load_serving_model("kan32", art, manifest("kan32", art, KAN32_FEATURES))
    assert model.horizons == (6, 12, 24)


def test_all_inputs_missing_predicts_nothing():
    art = single_bundle()
    model = load_serving_model("kan15", art, manifest("kan15", art, KAN15_FEATURES))
    assert model.predict(24, {"pm25_latest": None, "pm25_same_hour_24h": None}) is None


def test_feature_order_mismatch_fails_loudly():
    art = single_bundle()
    reordered = list(reversed(KAN15_FEATURES))
    with pytest.raises(ModelLoadError, match="do not match manifest feature_list"):
        load_serving_model("kan15", art, manifest("kan15", art, reordered))


def test_feature_unknown_to_shared_builder_fails_loudly():
    features = ["pm25_latest", "humidity"]
    art = single_bundle(features)
    with pytest.raises(ModelLoadError, match="not produced by the shared feature builder"):
        load_serving_model("x", art, manifest("x", art, features))


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"format": 2, "horizons": {}}, "unsupported model bundle format"),
        ({"format": 1}, "ambiguous model bundle"),
        ({"format": 1, "horizons": {"24": "x"}, "booster": "x"}, "ambiguous model bundle"),
        ({"format": 1, "horizon_hours": 48, "booster": "x"}, r"\+48h is not served"),
        ({"format": 1, "horizon_hours": 24, "booster": "not a model"}, "invalid LightGBM booster"),
    ],
)
def test_bad_bundles_are_rejected(payload, message):
    art = json.dumps(payload).encode()
    with pytest.raises(ModelLoadError, match=message):
        load_serving_model("x", art, manifest("x", art, KAN15_FEATURES))


def test_registry_lookup_and_integrity_errors_become_load_errors():
    reg = InMemoryRegistry()
    with pytest.raises(ModelLoadError, match="not found"):
        load_registered_model(reg, "missing")
    art = single_bundle()
    reg.rows["bad"] = ("0" * 64, art, json.dumps(manifest("bad", art, KAN15_FEATURES)))
    with pytest.raises(ModelLoadError, match="integrity"):
        load_registered_model(reg, "bad")


# --- predict step (shadow mode) ---------------------------------------------------------------


@pytest.fixture
def inputs_db():
    fake = FakeDb()
    load_inputs(fake)
    return fake


@pytest.fixture
def kan15_model():
    art = single_bundle()
    return load_serving_model("kan15-test", art, manifest("kan15-test", art, KAN15_FEATURES))


def test_challenger_rows_are_stored_next_to_baselines(inputs_db, monkeypatch, kan15_model):
    monkeypatch.setattr(predict, "load_registered_model", lambda con, v: kan15_model)
    result = do_run(inputs_db, challenger_model_version="kan15-test")
    assert not result.failures
    rows = inputs_db.predictions()
    challenger = [r for r in rows if r[3] == "kan15-test"]
    # one +24h row per station with inputs; 61t's only reading is 4h old, so it is skipped
    assert {r[0] for r in challenger} == {"59t"}
    assert any("61t: no inputs for kan15-test +24h" in m for m in result.skipped)
    assert all(r[2] == (ISSUE + timedelta(hours=24)).isoformat() for r in challenger)
    assert result.per_model["kan15-test"] == len(challenger)
    assert result.per_model[PERSISTENCE_MODEL_VERSION] > 0


def test_no_challenger_configured_keeps_baseline_behaviour(inputs_db, monkeypatch):
    def unexpected(*_a):
        raise AssertionError("registry must not be read without a challenger")

    monkeypatch.setattr(predict, "load_registered_model", unexpected)
    result = do_run(inputs_db, challenger_model_version="  ")
    assert not result.failures
    assert set(result.per_model) == {PERSISTENCE_MODEL_VERSION, CAMS_PASSTHROUGH_MODEL_VERSION}


def test_unservable_challenger_fails_loudly_but_baselines_are_written(inputs_db, monkeypatch):
    def broken(con, version):
        raise ModelLoadError("+24h booster features do not match manifest feature_list")

    monkeypatch.setattr(predict, "load_registered_model", broken)
    result = do_run(inputs_db, challenger_model_version="kan15-test")
    assert result.failures and "kan15-test not served" in result.failures[0]
    assert result.rows_written > 0
    assert all(r[3] != "kan15-test" for r in inputs_db.predictions())


def test_challenger_name_cannot_shadow_a_baseline(inputs_db):
    with pytest.raises(predict.ConfigError, match="must differ"):
        do_run(inputs_db, challenger_model_version=PERSISTENCE_MODEL_VERSION)


# --- registry CLI ------------------------------------------------------------------------------


def cli(reg, *argv):
    lines = []
    code = registry_cli.run(
        list(argv), database_url="db", connect=lambda _url: reg, out=lines.append
    )
    return code, "\n".join(lines)


def test_cli_register_check_and_manual_promote(tmp_path):
    reg = InMemoryRegistry()
    art = single_bundle()
    (tmp_path / "m.json").write_bytes(art)
    (tmp_path / "man.json").write_text(json.dumps(manifest("kan15", art, KAN15_FEATURES)))
    code, text = cli(reg, "register", "--artifact", str(tmp_path / "m.json"),
                     "--manifest", str(tmp_path / "man.json"))  # fmt: skip
    assert code == 0 and "champion unchanged" in text and reg.champion is None
    assert cli(reg, "check", "kan15") == (
        0,
        "ok kan15: horizons=[24] features=" + str(KAN15_FEATURES),
    )
    assert cli(reg, "champion") == (0, "no production champion is set")
    code, text = cli(reg, "promote", "kan15")
    assert code == 1 and "--confirm" in text and reg.champion is None
    assert cli(reg, "promote", "kan15", "--confirm") == (0, "production champion: none -> kan15")
    assert cli(reg, "champion") == (0, "production champion: kan15")


def test_cli_refuses_to_promote_an_unservable_model():
    reg = InMemoryRegistry()
    features = ["pm25_latest", "humidity"]
    art = single_bundle(features)
    reg.rows["bad"] = (
        registry.artifact_sha256(art),
        art,
        json.dumps(manifest("bad", art, features)),
    )
    code, text = cli(reg, "promote", "bad", "--confirm")
    assert code == 1 and "shared feature builder" in text and reg.champion is None


def test_model_trained_with_zero_reporting_lag_is_refused():
    art = single_bundle()
    man = {**manifest("kan15", art, KAN15_FEATURES), "assumed_reporting_lag_hours": 0}
    with pytest.raises(ModelLoadError, match="assumed_reporting_lag_hours=0"):
        load_serving_model("kan15", art, man)
    man["assumed_reporting_lag_hours"] = 1
    assert load_serving_model("kan15", art, man).horizons == (24,)


def test_partial_single_horizon_bundle_is_rejected_up_front():
    art = json.dumps({"format": 1, "horizon_hours": 24}).encode()
    with pytest.raises(ModelLoadError, match="needs both horizon_hours and booster"):
        load_serving_model("x", art, manifest("x", art, KAN15_FEATURES))


def test_non_finite_prediction_is_an_error(kan15_model, monkeypatch):
    booster = kan15_model.boosters[24]
    monkeypatch.setattr(booster, "predict", lambda _x: np.asarray([np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        kan15_model.predict(24, {"pm25_latest": 1.0, "pm25_same_hour_24h": 1.0})


def test_challenger_crash_drops_all_challenger_rows_and_keeps_baselines(
    inputs_db, monkeypatch, kan15_model
):
    def boom(_x):
        raise lgb.basic.LightGBMError("boom")

    monkeypatch.setattr(kan15_model.boosters[24], "predict", boom)
    monkeypatch.setattr(predict, "load_registered_model", lambda con, v: kan15_model)
    result = do_run(inputs_db, challenger_model_version="kan15-test")
    assert any("dropped for this run" in m for m in result.failures)
    rows = inputs_db.predictions()
    assert rows and all(r[3] != "kan15-test" for r in rows)
    assert result.per_model.get("kan15-test", 0) == 0
