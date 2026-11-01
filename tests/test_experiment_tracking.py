"""KAN-81: tracking helpers. No network; a temporary local store."""

import hashlib
import json

import pytest

from pm25_forecast.experiments import tracking

SHA = "a" * 64


def tags(**over):
    base = dict.fromkeys(tracking.REQUIRED_TAGS, "x")
    base.update(ticket="KAN-85", split="dev", git_dirty="false", snapshot_sha256=SHA, prereg="p-1")
    base.update(over)
    return base


def caller_tags(**over):
    base = {k: v for k, v in tags().items() if k not in tracking.AUTO_TAGS}
    base.update(over)
    return base


GOOD = {"mae_24h": 5.0, "rmse_24h": 7.0, "n_pairs_24h": 100}
EXP = "kan78/e00-test"


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No MLFLOW_* in the process, and no .env reachable from cwd or the (patched) repo root."""
    for name in (tracking.STORE_ENV, tracking.ARTIFACT_ENV):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "cwd").mkdir(exist_ok=True)
    (tmp_path / "repo").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path / "repo")
    return tmp_path


def test_default_store_is_shared_sqlite(clean_env):
    assert "/pm25-experiments/mlflow/mlflow.db?timeout=" in tracking.tracking_uri()
    assert tracking.tracking_uri().startswith("sqlite:///")


def test_env_overrides_store_and_gets_busy_timeout(clean_env, monkeypatch):
    monkeypatch.setenv(tracking.STORE_ENV, "sqlite:////tmp/x.db")
    assert tracking.tracking_uri() == "sqlite:////tmp/x.db?timeout=30"
    monkeypatch.setenv(tracking.STORE_ENV, "sqlite:////tmp/x.db?timeout=5")
    assert tracking.tracking_uri() == "sqlite:////tmp/x.db?timeout=5"
    monkeypatch.setenv(tracking.STORE_ENV, "  ")
    assert "pm25-experiments" in tracking.tracking_uri()


def test_dotenv_is_read_and_process_env_wins(clean_env, monkeypatch):
    (clean_env / "cwd" / ".env").write_text("MLFLOW_TRACKING_URI=sqlite:////tmp/dot env/a.db\n")
    assert tracking.tracking_uri() == "sqlite:////tmp/dot env/a.db?timeout=30"
    monkeypatch.setenv(tracking.STORE_ENV, "sqlite:////tmp/b.db")
    assert tracking.tracking_uri().startswith("sqlite:////tmp/b.db")


def test_repo_root_dotenv_is_read(clean_env):
    (clean_env / "repo" / ".env").write_text(f"{tracking.ARTIFACT_ENV}={clean_env}/arts\n")
    assert tracking.artifact_root() == (clean_env / "arts").as_uri()


def test_artifact_root_next_to_db_or_explicit(clean_env, monkeypatch):
    monkeypatch.setenv(tracking.STORE_ENV, f"sqlite:///{clean_env / 'my store' / 'm.db'}")
    assert tracking.artifact_root() == (clean_env / "my store" / "artifacts").as_uri()
    monkeypatch.setenv(tracking.ARTIFACT_ENV, str(clean_env / "elsewhere"))
    assert tracking.artifact_root() == (clean_env / "elsewhere").as_uri()
    monkeypatch.delenv(tracking.ARTIFACT_ENV)
    monkeypatch.setenv(tracking.STORE_ENV, "postgresql://u@h/db")
    with pytest.raises(tracking.TrackingError, match=tracking.ARTIFACT_ENV):
        tracking.artifact_root()


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


def test_git_state_counts_untracked_but_not_output_dirs(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@x", *a], cwd=tmp_path, check=True
        )

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    assert tracking.git_state(tmp_path)[1] is False
    for d in ("report", ".claude", "out"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.html").write_text("x")
    assert tracking.git_state(tmp_path)[1] is False
    (tmp_path / "new_module.py").write_text("y = 2\n")
    assert tracking.git_state(tmp_path)[1] is True


def test_git_state_shape():
    sha, dirty = tracking.git_state()
    assert len(sha) == 40 and isinstance(dirty, bool)


@pytest.fixture
def store(clean_env, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'my store' / 'm.db'}"
    monkeypatch.setenv(tracking.STORE_ENV, uri)
    # mlflow's tracking URI is process-global: pin it to this test's store, otherwise runs()
    # reads whichever store an earlier test left behind when log_run refuses before connecting.
    (tmp_path / "my store").mkdir()
    mlflow.set_tracking_uri(uri)
    # Independent of the checkout's state: holdout runs need a clean tree.
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))
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
    assert run.data.tags["git_sha"] == "c" * 40
    assert run.data.tags["git_dirty"] == "false"
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
    "prereg": "p1",
}


def test_render_leaderboard_formats():
    md = tracking.render_leaderboard([ROW], "md")
    assert "| experiment | run | mae_24h |" in md
    assert "| kan78/e1 | a (12345678) | 5.000 | 7.000 |  | 90 | abc | def | DIRTY | p1 |" in md
    wiki = tracking.render_leaderboard([ROW], "confluence")
    assert wiki.startswith("||experiment||run||mae_24h||")
    assert "|kan78/e1|" in wiki and "DIRTY" in wiki
    with pytest.raises(tracking.TrackingError):
        tracking.render_leaderboard([], "csv")
    assert "WARNING" not in md


def test_leaderboard_warns_on_mixed_snapshots_or_prereg():
    other = {**ROW, "experiment": "kan78/e2", "snapshot": "zzz", "prereg": "p2"}
    for fmt in ("md", "confluence"):
        text = tracking.render_leaderboard([ROW, other], fmt)
        assert "different snapshot_sha256" in text and "different prereg" in text


def test_leaderboard_best_run_dev_only_flags_invalid(store, monkeypatch):
    import mlflow

    log = tracking.log_run
    log("kan78/e1", "worse", {}, {**GOOD, "mae_24h": 6.0}, caller_tags())
    log("kan78/e1", "best", {}, {**GOOD, "mae_24h": 4.0}, caller_tags())
    bad = log("kan78/e1", "bad", {}, {**GOOD, "mae_24h": 1.0}, caller_tags())
    mlflow.MlflowClient().set_tag(bad, "invalid", "true")
    better = {**GOOD, "mae_24h": 3.0, "skill_vs_persistence_24h": 0.1}
    monkeypatch.setattr(tracking, "git_state", lambda: ("d" * 40, True))
    log("kan78/e2", "only", {}, better, caller_tags())  # dev run from a dirty tree
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))
    log("kan78/e3", "ho", {}, {**GOOD, "mae_24h": 2.0}, caller_tags(split="holdout"))

    rows = tracking.collect_leaderboard()
    assert [r["experiment"] for r in rows] == ["kan78/e2", "kan78/e1"]
    assert rows[1]["run"].startswith("best ") and rows[1]["mae"] == 4.0
    assert rows[0]["skill"] == 0.1
    assert rows[0]["dirty"] is True and rows[1]["dirty"] is False
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


def test_experiment_artifacts_go_to_the_shared_store_not_cwd(store):
    import mlflow

    tracking.log_run(EXP, "r", {}, GOOD, caller_tags())
    exp = mlflow.MlflowClient().get_experiment_by_name(EXP)
    assert exp.artifact_location == (store / "my store" / "artifacts").as_uri()
    assert not (store / "cwd" / "mlruns").exists()


def test_invalid_tag_and_missing_snapshot_refused(store):
    with pytest.raises(tracking.TrackingError, match="invalid tag"):
        tracking.log_run(EXP, "r", {}, GOOD, {**caller_tags(), "invalid": "false"})
    t = caller_tags()
    del t["snapshot_sha256"]
    with pytest.raises(tracking.TrackingError, match="snapshot_sha256 not given"):
        tracking.log_run(EXP, "r", {}, GOOD, t)
    (store / "m.json").write_text("{}")
    with pytest.raises(tracking.TrackingError, match="no valid snapshot_sha256"):
        tracking.log_run(EXP, "r", {}, GOOD, caller_tags(snapshot_sha256=str(store / "m.json")))
    assert runs() == []


def test_lock_error_mid_write_resumes_same_run(store, monkeypatch):
    import mlflow

    monkeypatch.setattr(tracking.time, "sleep", lambda _s: None)
    original = mlflow.MlflowClient.log_batch
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mlflow.MlflowClient, "log_batch", flaky)
    run_id = tracking.log_run(EXP, "r", {"a": 1}, GOOD, caller_tags())
    (run,) = runs()
    assert run.info.run_id == run_id and run.info.status == "FINISHED"
    assert run.data.metrics["mae_24h"] == 5.0


def test_failed_write_marks_run_failed_and_leaderboard_skips_it(store, monkeypatch):
    import mlflow

    def boom(self, *args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(mlflow.MlflowClient, "log_batch", boom)
    with pytest.raises(RuntimeError, match="disk"):
        tracking.log_run(EXP, "r", {}, GOOD, caller_tags())
    (run,) = runs()
    assert run.info.status == "FAILED"
    assert tracking.collect_leaderboard() == []


def test_holdout_md_leaderboard_must_go_outside_repo(store, capsys):
    repo = store / "repo"
    tracking.log_run("kan78/e1", "a", {}, GOOD, caller_tags(split="holdout"))
    with pytest.raises(tracking.TrackingError, match="outside the repository"):
        tracking.main(["leaderboard", "--holdout", "--format", "md"])
    with pytest.raises(tracking.TrackingError, match="outside the repository"):
        tracking.main(["leaderboard", "--holdout", "--format", "md", "--out", str(repo / "x.md")])
    out = store / "holdout.md"
    assert tracking.main(["leaderboard", "--holdout", "--format", "md", "--out", str(out)]) == 0
    assert "kan78/e1" in out.read_text()
    assert "WARNING: holdout" in capsys.readouterr().err
