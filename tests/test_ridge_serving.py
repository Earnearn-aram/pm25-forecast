"""KAN-100: pooled ridge challenger: train/serve parity, serving lag, gaps, registry, round trip."""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments import linear_models as lm
from pm25_forecast.models import ridge
from pm25_forecast.models.ridge_bundle import parse_ridge_bundle
from pm25_forecast.models.serving import (
    ModelLoadError,
    RidgeModel,
    load_registered_model,
    load_ridge_model,
)
from pm25_forecast.storage import registry
from test_model_registry import InMemoryRegistry
from test_serving import cli

T0 = datetime(2026, 8, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
N = 600
HORIZONS = (1, 6, 24)
STATIONS = ("a", "b")
DATA_SHA = hashlib.sha256(b"fixture").hexdigest()
REFIT = T0 + N * HOUR


def series(station: str, seed: int) -> list[Observation]:
    rng = np.random.default_rng(seed)
    out, level = [], 30.0
    for i in range(N + 60):  # a little beyond the refit time: unmatured rows must not be trained on
        level = 0.9 * level + 3.0 + rng.normal(0, 2.0)
        ts = T0 + i * HOUR
        out.append(Observation(station, ts, max(level, 1.0), ts + HOUR))
    return out


OBS = [o for k, s in enumerate(STATIONS) for o in series(s, k)]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    import os

    os.environ["PM25_KAN84_LOG_DIR"] = str(tmp_path_factory.mktemp("kan84"))
    lm._dumps.clear()
    train = ridge.train_snapshot(OBS, REFIT, HORIZONS)
    out = {}
    for policy in ("skip", "train_mean"):
        section, _ = ridge.fit_bundle(train, policy)
        artifact = ridge.ridge_artifact(section)
        manifest = ridge.ridge_manifest(
            artifact,
            section,
            version=f"ridge-test-{policy}",
            window=(T0, REFIT),
            data_sha256=DATA_SHA,
            requested=HORIZONS,
            dev_mae={"1": {"ridge": 2.0, "persistence": 1.5, "smoother": 1.7}},
        )
        out[policy] = (artifact, manifest)
    experiment = lm.fit(train, dict(ridge.VARIANT_PARAMS))  # the experiment's own factory
    return train, out, experiment


def model(fitted, policy="train_mean") -> RidgeModel:
    artifact, manifest = fitted[1][policy]
    return load_ridge_model(manifest["version"], artifact, manifest)


def history(station, origin, drop=(), extra=()):
    rows = [o for o in OBS if o.station_id == station and o.ts < origin]
    return [o for o in rows if o.ts not in {origin - k * HOUR for k in drop}] + list(extra)


ORIGINS = [REFIT + k * HOUR for k in (0, 5, 17)]


@pytest.mark.parametrize("policy", ["skip", "train_mean"])
def test_train_serve_parity_with_experiment_factory(fitted, policy):
    experiment = fitted[2]
    ridge_model = model(fitted, policy)
    checked = 0
    for station in STATIONS:
        for origin in ORIGINS:
            obs = history(station, origin)
            got = ridge_model.forecast(obs, station, origin)
            for h in HORIZONS:
                want = experiment(Snapshot(origin, station, tuple(obs), ()), h)
                assert want is not None and got[h] is not None
                assert got[h] == pytest.approx(max(0.0, want), abs=1e-9)
                checked += 1
    assert checked == len(STATIONS) * len(ORIGINS) * len(HORIZONS)


def test_train_mean_policy_matches_experiment_on_a_gap_and_skip_returns_none(fitted):
    origin = ORIGINS[0]
    obs = history("a", origin, drop=(2,))  # lag_2h missing; the latest (lag_1h) is still there
    want = fitted[2](Snapshot(origin, "a", tuple(obs), ()), 6)
    assert want is not None  # the experiment imputes the training mean for the missing lag
    assert model(fitted, "train_mean").forecast(obs, "a", origin)[6] == pytest.approx(
        max(0.0, want), abs=1e-9
    )
    assert model(fitted, "skip").forecast(obs, "a", origin) == dict.fromkeys(HORIZONS)


def test_no_latest_reading_or_unknown_station_gives_no_forecast(fitted):
    origin = ORIGINS[0]
    for policy in ("skip", "train_mean"):
        m = model(fitted, policy)
        stale = history("a", origin, drop=(1, 2, 3))
        assert m.forecast(stale, "a", origin) == dict.fromkeys(HORIZONS)
        assert m.forecast(history("a", origin), "zzz", origin) == dict.fromkeys(HORIZONS)


def test_predict_requires_every_feature_key(fitted):
    with pytest.raises(ModelLoadError, match="did not produce"):
        model(fitted).predict(6, {"pm25_lag_1h": 5.0}, "a")


def test_serving_lag_ignores_readings_at_or_after_the_origin_hour_and_unavailable_rows(fitted):
    origin = ORIGINS[1]
    m = model(fitted)
    clean = m.forecast(history("a", origin), "a", origin)
    poison = [
        # ts at and after the origin: never visible
        *(Observation("a", origin + k * HOUR, 1000.0, origin + k * HOUR) for k in range(0, 5)),
        # an older hour re-delivered later than the origin: available_at > origin, ignored
        Observation("a", origin - HOUR, 1000.0, origin + HOUR),
        Observation("a", origin - 5 * HOUR, 1000.0, origin + 3 * HOUR),
    ]
    assert m.forecast(history("a", origin, extra=poison), "a", origin) == clean
    # the newest allowed reading is origin - 1 h and it is used
    changed = [
        Observation(o.station_id, o.ts, o.value + 50 if o.ts == origin - HOUR else o.value,
                    o.available_at)
        for o in history("a", origin)
    ]  # fmt: skip
    assert m.forecast(changed, "a", origin)[1] != clean[1]


def test_forecast_rejects_a_naive_origin(fitted):
    with pytest.raises(ValueError):
        model(fitted).forecast([], "a", datetime(2026, 9, 1))  # noqa: DTZ001


def test_bundle_round_trip_and_determinism(fitted):
    artifact, manifest = fitted[1]["train_mean"]
    spec = parse_ridge_bundle(json.loads(artifact))
    assert tuple(spec.horizons) == HORIZONS and spec.stations == STATIONS
    assert len(spec.horizons[24].coef) == len(spec.features) + len(spec.stations)
    section = json.loads(artifact)["ridge"]
    assert ridge.ridge_artifact(section) == artifact
    assert manifest["assumed_reporting_lag_hours"] == 1
    assert manifest["feature_list"] == list(spec.features)
    assert manifest["horizons"] == list(HORIZONS)
    assert manifest["horizons_worse_than_persistence_dev"] == [1]
    assert manifest["artifact_sha256"] == hashlib.sha256(artifact).hexdigest()


def test_unmatured_rows_are_not_trained_on():
    train = ridge.train_snapshot(OBS, REFIT, HORIZONS)
    assert max(o.ts for o in train.observations) <= REFIT
    assert all(o.available_at <= REFIT for o in train.observations)


def test_registry_accepts_the_artifact_and_cli_check_loads_it(fitted, tmp_path):
    artifact, manifest = fitted[1]["train_mean"]
    reg = InMemoryRegistry()
    (tmp_path / "a.json").write_bytes(artifact)
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    code, text = cli(reg, "register", "--artifact", str(tmp_path / "a.json"),
                     "--manifest", str(tmp_path / "m.json"))  # fmt: skip
    assert code == 0, text
    code, text = cli(reg, "check", manifest["version"])
    assert code == 0 and "horizons=[1, 6, 24]" in text
    assert isinstance(load_registered_model(reg, manifest["version"]), RidgeModel)


def test_registry_and_loader_stay_strict(fitted):
    artifact, manifest = fitted[1]["train_mean"]
    reg = InMemoryRegistry()
    bad_features = {**manifest, "feature_list": manifest["feature_list"][::-1]}
    with pytest.raises(ValueError, match="feature_list"):
        registry.write_model(reg, manifest["version"], artifact, bad_features)
    with pytest.raises(ValueError, match="artifact_sha256"):
        registry.write_model(reg, manifest["version"], artifact + b" ", manifest)
    with pytest.raises(ModelLoadError, match="assumed_reporting_lag_hours"):
        load_ridge_model("v", artifact, {**manifest, "assumed_reporting_lag_hours": 0})
    with pytest.raises(ModelLoadError, match="features do not match"):
        load_ridge_model("v", artifact, {**manifest, "feature_list": ["pm25_lag_1h"]})
    section = json.loads(artifact)["ridge"]
    for broken in (
        {**section, "features": [*section["features"][:-1], "not_a_feature"]},
        {**section, "horizons": {"25": section["horizons"]["24"]}},
        {**section, "output": "level"},
        {**section, "missing_policy": "zero"},
        {**section, "stations": ["a"]},  # coefficient length no longer matches
    ):
        raw = json.dumps({"format": 1, "ridge": broken}).encode()
        with pytest.raises(ModelLoadError):
            load_ridge_model("v", raw, manifest)


def test_default_horizons_are_every_hour():
    assert list(ridge.DEFAULT_HORIZONS) == list(range(1, 25))


def test_cli_writes_artifact_and_manifest(fitted, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ridge, "load_observations", lambda snapshot, kan74_repaired: (OBS, DATA_SHA)
    )
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"6": {"ridge": 1.0, "persistence": 2.0, "smoother": 1.5}}))
    code = ridge.main(
        ["--out-dir", str(tmp_path / "out"), "--horizons", "6", "24", "--as-of",
         REFIT.isoformat(), "--version", "ridge-cli", "--dev-mae", str(dev)]
    )  # fmt: skip
    assert code == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    artifact = (tmp_path / "out" / "artifact.json").read_bytes()
    registry.validate_manifest(manifest)
    assert manifest["horizons"] == [6, 24] and manifest["data_sha256"] == DATA_SHA
    assert manifest["horizons_worse_than_persistence_dev"] == []
    assert manifest["training_window"]["end"] == REFIT.isoformat()
    reg = InMemoryRegistry()
    registry.write_model(reg, "ridge-cli", artifact, manifest)


def test_serving_import_needs_no_experiments_code():
    code = (
        "import sys, lightgbm;"  # lightgbm itself imports scikit-learn when installed
        "before = set(sys.modules);"
        "import pm25_forecast.models.serving, pm25_forecast.registry_cli;"
        "bad = [m for m in set(sys.modules) - before if m.startswith(('sklearn', 'statsmodels',"
        " 'mlflow', 'pm25_forecast.experiments'))];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_2027_holiday_gap_is_imputed_and_logged_under_train_mean(fitted, caplog):
    origin = datetime(2027, 1, 2, 3, tzinfo=UTC)
    obs = [
        Observation("a", origin - k * HOUR, 30.0 + k % 5, origin - (k - 1) * HOUR)
        for k in range(1, 80)
    ]
    with caplog.at_level(logging.WARNING, logger="pm25_forecast.models.serving"):
        got = model(fitted, "train_mean").forecast(obs, "a", origin)
    assert all(v is not None for v in got.values())
    assert any("is_thai_holiday" in r.getMessage() for r in caplog.records)
    assert model(fitted, "skip").forecast(obs, "a", origin) == dict.fromkeys(HORIZONS)


def test_more_than_a_quarter_of_features_missing_gives_no_forecast(fitted):
    origin = ORIGINS[0]
    m = model(fitted)
    five = history("a", origin, drop=(2, 3, 6, 12))  # lag_2/3/6/12h + diff_1h = 5 of 20: allowed
    assert all(v is not None for v in m.forecast(five, "a", origin).values())
    six = history("a", origin, drop=(2, 3, 6, 12, 24))  # 6 of 20 missing: refused
    assert m.forecast(six, "a", origin) == dict.fromkeys(HORIZONS)


def test_holiday_table_coverage_warning():
    assert ridge.holiday_table_warning(datetime(2026, 10, 1, tzinfo=UTC)) is None
    assert "KAN-106" in ridge.holiday_table_warning(datetime(2026, 11, 20, tzinfo=UTC))


def test_loader_rejects_a_manifest_without_the_lag(fitted):
    artifact, manifest = fitted[1]["train_mean"]
    without = {k: v for k, v in manifest.items() if k != "assumed_reporting_lag_hours"}
    with pytest.raises(ModelLoadError, match="assumed_reporting_lag_hours"):
        load_ridge_model("v", artifact, without)


def test_huge_numbers_in_a_bundle_are_a_clean_error(fitted):
    artifact, manifest = fitted[1]["train_mean"]
    section = json.loads(artifact)["ridge"]
    section["horizons"]["24"]["intercept"] = 10**400
    raw = json.dumps({"format": 1, "ridge": section}).encode()
    with pytest.raises(ModelLoadError):
        load_ridge_model("v", raw, manifest)
