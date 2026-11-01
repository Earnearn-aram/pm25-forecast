"""KAN-81: tracking helpers. No network; a temporary local store."""

import hashlib

import pytest

from pm25_forecast.experiments import tracking

SHA = "a" * 64


def tags(**over):
    base = {t: "x" for t in tracking.REQUIRED_TAGS}
    base.update(
        ticket="KAN-85", split="dev", git_dirty="false", snapshot_sha256=SHA, prereg="p-1"
    )
    base.update(over)
    return base


GOOD = {"mae_24h": 5.0, "rmse_24h": 7.0, "n_pairs_24h": 100}


def test_default_store_is_shared_sqlite(monkeypatch):
    monkeypatch.delenv(tracking.STORE_ENV, raising=False)
    assert tracking.tracking_uri().endswith("/pm25-experiments/mlflow/mlflow.db")
    assert tracking.tracking_uri().startswith("sqlite:///")


def test_env_overrides_store(monkeypatch):
    monkeypatch.setenv(tracking.STORE_ENV, "sqlite:///tmp/x.db")
    assert tracking.tracking_uri() == "sqlite:///tmp/x.db"
    monkeypatch.setenv(tracking.STORE_ENV, "  ")
    assert "pm25-experiments" in tracking.tracking_uri()


def test_valid_tags_pass():
    assert tracking.validate_tags(tags())["split"] == "dev"


@pytest.mark.parametrize("name", tracking.REQUIRED_TAGS)
def test_missing_tag_rejected(name):
    t = tags()
    del t[name]
    with pytest.raises(tracking.TrackingError, match=name):
        tracking.validate_tags(t)


@pytest.mark.parametrize(
    "over",
    [
        {"split": "test"},
        {"ticket": "85"},
        {"snapshot_sha256": "abc"},
        {"git_dirty": "maybe"},
        {"git_dirty": "true", "split": "holdout"},
        {"surprise": "1"},
    ],
)
def test_bad_tags_rejected(over):
    with pytest.raises(tracking.TrackingError):
        tracking.validate_tags(tags(**over))


def test_dirty_dev_allowed():
    assert tracking.validate_tags(tags(git_dirty="True"))["git_dirty"] == "true"


def test_metrics_drop_undefined_and_keep_names():
    out = tracking.validate_metrics({**GOOD, "spike_recall_24h": float("nan"), "mae_6h": None})
    assert "spike_recall_24h" not in out and "mae_6h" not in out
    assert out["mae_24h"] == 5.0


def test_metrics_reject_unknown_and_missing():
    with pytest.raises(tracking.TrackingError, match="unknown"):
        tracking.validate_metrics({**GOOD, "mae24": 1.0})
    with pytest.raises(tracking.TrackingError, match="mae_24h"):
        tracking.validate_metrics({"rmse_24h": 1.0, "n_pairs_24h": 3})


def test_file_sha256(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert tracking.file_sha256(p) == hashlib.sha256(b"abc").hexdigest()


def test_git_state_shape():
    sha, dirty = tracking.git_state()
    assert len(sha) == 40 and isinstance(dirty, bool)


@pytest.fixture
def store(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.setenv(tracking.STORE_ENV, f"sqlite:///{tmp_path / 'my store' / 'm.db'}")
    return tmp_path


def test_tracked_run_roundtrip(store):
    import mlflow

    with tracking.tracked_run("r1", tags=tags(), params={"seed": 1}) as m:
        m.update(GOOD)
        m["spike_recall_24h"] = float("nan")
    run = mlflow.search_runs(
        experiment_names=[tracking.DEFAULT_EXPERIMENT], output_format="list"
    )[0]
    assert run.data.tags["prereg"] == "p-1"
    assert run.data.metrics["mae_24h"] == 5.0
    assert "spike_recall_24h" not in run.data.metrics
    assert run.data.params["seed"] == "1"
    assert "invalid" not in run.data.tags


def test_failed_run_tagged_invalid(store):
    import mlflow

    with pytest.raises(RuntimeError), tracking.tracked_run("r2", tags=tags(), params={}):
        raise RuntimeError("boom")
    run = mlflow.search_runs(
        experiment_names=[tracking.DEFAULT_EXPERIMENT], output_format="list"
    )[0]
    assert run.data.tags["invalid"] == "true"


def test_missing_metric_tagged_invalid(store):
    import mlflow

    with pytest.raises(tracking.TrackingError), tracking.tracked_run("r3", tags=tags(), params={}):
        pass
    run = mlflow.search_runs(
        experiment_names=[tracking.DEFAULT_EXPERIMENT], output_format="list"
    )[0]
    assert run.data.tags["invalid"] == "true"
