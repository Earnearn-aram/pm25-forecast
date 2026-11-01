"""KAN-81: tracking helpers. No network; a temporary local store."""

import hashlib
import json

import pytest

from pm25_forecast.experiments import tracking

SHA = "a" * 64


def tags(**over):
    base = {t: "x" for t in tracking.REQUIRED_TAGS}
    base.update(ticket="KAN-85", split="dev", git_dirty="false", snapshot_sha256=SHA, prereg="p-1")
    base.update(over)
    return base


def caller_tags(**over):
    base = {k: v for k, v in tags().items() if k not in tracking.AUTO_TAGS}
    base.update(over)
    return base


GOOD = {"mae_24h": 5.0, "rmse_24h": 7.0, "n_pairs_24h": 100}
EXP = "kan78/e00-test"


def test_default_store_is_shared_sqlite(monkeypatch):
    monkeypatch.delenv(tracking.STORE_ENV, raising=False)
    assert "/pm25-experiments/mlflow/mlflow.db?timeout=" in tracking.tracking_uri()
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


def test_n_pairs_is_optional():
    assert tracking.validate_metrics({"mae_24h": 1.0, "rmse_24h": 2.0})["mae_24h"] == 1.0


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


def runs(name=EXP):
    import mlflow

    return mlflow.search_runs(experiment_names=[name], output_format="list")


def test_valid_run_logged(store):
    art = store / "note.txt"
    art.write_text("hi")
    run_id = tracking.log_run(
        EXP,
        "r1",
        params={"seed": 1},
        metrics={**GOOD, "spike_recall_24h": float("nan")},
        tags=caller_tags(),
        artifacts=[art],
    )
    (run,) = runs()
    assert run.info.run_id == run_id and run.info.status == "FINISHED"
    assert run.data.tags["prereg"] == "p-1"
    assert len(run.data.tags["git_sha"]) == 40
    assert run.data.tags["git_dirty"] in ("true", "false")
    assert run.data.metrics["mae_24h"] == 5.0
    assert "spike_recall_24h" not in run.data.metrics
    assert run.data.params["seed"] == "1"
    import mlflow

    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == ["note.txt"]


def test_snapshot_hash_from_e02_manifest(store):
    snap = store / "snap"
    snap.mkdir()
    (snap / "manifest.json").write_text(json.dumps({"snapshot_sha256": "b" * 64}))
    tracking.log_run(EXP, "r", {}, GOOD, caller_tags(snapshot_sha256=str(snap)))
    assert runs()[0].data.tags["snapshot_sha256"] == "b" * 64


@pytest.mark.parametrize("name", ["ticket", "prereg", "split", "snapshot_sha256", "agent"])
def test_run_missing_tag_refused_and_nothing_written(store, name):
    t = caller_tags()
    del t[name]
    with pytest.raises(tracking.TrackingError, match=name):
        tracking.log_run(EXP, "bad", {}, GOOD, t)
    assert runs() == []


def test_run_missing_metric_refused(store):
    with pytest.raises(tracking.TrackingError, match="rmse_24h"):
        tracking.log_run(EXP, "bad", {}, {"mae_24h": 1.0}, caller_tags())
    assert runs() == []


def test_manual_git_tags_and_bad_names_refused(store):
    with pytest.raises(tracking.TrackingError, match="automatically"):
        tracking.log_run(EXP, "r", {}, GOOD, {**caller_tags(), "git_sha": "x"})
    with pytest.raises(tracking.TrackingError, match="experiment"):
        tracking.log_run("other/e1", "r", {}, GOOD, caller_tags())
    with pytest.raises(tracking.TrackingError, match="artifacts not found"):
        tracking.log_run(EXP, "r", {}, GOOD, caller_tags(), [store / "nope"])


def test_two_writers_at_once(store):
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        return tracking.log_run(EXP, f"r{i}", {}, {**GOOD, "mae_24h": float(i + 1)}, caller_tags())

    with ThreadPoolExecutor(4) as pool:
        ids = list(pool.map(one, range(4)))
    assert len(set(ids)) == 4 and len(runs()) == 4


def test_lock_errors_are_retried(monkeypatch):
    monkeypatch.setattr(tracking.time, "sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return "ok"

    assert tracking._with_retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(ValueError):
        tracking._with_retry(lambda: (_ for _ in ()).throw(ValueError("other")))


ROW = {
    "experiment": "kan78/e1",
    "run": "a (12345678)",
    "mae": 5.0,
    "rmse": 7.0,
    "skill": None,
    "n_pairs": 90,
    "git_sha": "abc",
    "snapshot": "def",
    "dirty": True,
}


def test_render_leaderboard_formats():
    md = tracking.render_leaderboard([ROW], "md")
    assert "| experiment | run | mae_24h |" in md
    assert "| kan78/e1 | a (12345678) | 5.000 | 7.000 |  | 90 | abc | def | DIRTY |" in md
    wiki = tracking.render_leaderboard([ROW], "confluence")
    assert wiki.startswith("||experiment||run||mae_24h||")
    assert "|kan78/e1|" in wiki and "DIRTY" in wiki
    with pytest.raises(tracking.TrackingError):
        tracking.render_leaderboard([], "csv")


def test_leaderboard_best_run_dev_only_flags_invalid(store):
    import mlflow

    log = tracking.log_run
    log("kan78/e1", "worse", {}, {**GOOD, "mae_24h": 6.0}, caller_tags())
    log("kan78/e1", "best", {}, {**GOOD, "mae_24h": 4.0}, caller_tags())
    bad = log("kan78/e1", "bad", {}, {**GOOD, "mae_24h": 1.0}, caller_tags())
    mlflow.MlflowClient().set_tag(bad, "invalid", "true")
    better = {**GOOD, "mae_24h": 3.0, "skill_vs_persistence_24h": 0.1}
    log("kan78/e2", "only", {}, better, caller_tags())
    log("kan78/e3", "ho", {}, {**GOOD, "mae_24h": 2.0}, caller_tags(split="holdout"))

    rows = tracking.collect_leaderboard()
    assert [r["experiment"] for r in rows] == ["kan78/e2", "kan78/e1"]
    assert rows[1]["run"].startswith("best ") and rows[1]["mae"] == 4.0
    assert rows[0]["skill"] == 0.1
    assert rows[0]["dirty"] is tracking.git_state()[1]
    assert [r["experiment"] for r in tracking.collect_leaderboard(split="holdout")] == ["kan78/e3"]


def test_leaderboard_cli_writes_md(store, capsys):
    tracking.log_run("kan78/e1", "a", {}, GOOD, caller_tags())
    out = store / "lb.md"
    assert tracking.main(["leaderboard", "--format", "md", "--out", str(out)]) == 0
    assert "kan78/e1" in out.read_text()
    assert tracking.main(["leaderboard", "--format", "confluence"]) == 0
    assert "||experiment||" in capsys.readouterr().out


def test_demo_persistence_run_logged(store):
    run_id = tracking._demo()
    (run,) = runs("kan78/e00-demo")
    assert run.info.run_id == run_id
    assert run.data.metrics["n_pairs_24h"] == 72
    assert run.data.metrics["mae_24h"] > 0


def test_render_report_attaches_html(store):
    pytest.importorskip("nbconvert")
    import mlflow

    run_id = tracking.log_run(EXP, "r", {}, GOOD, caller_tags())
    nb = store / "nb.ipynb"
    nb.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# hi"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    html = tracking.render_report(nb, run_id=run_id, out_dir=store / "report")
    assert html.name == "report.html" and html.is_file()
    names = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "report")]
    assert names == ["report/report.html"]
