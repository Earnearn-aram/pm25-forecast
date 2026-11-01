"""KAN-97: HTML experiment reports. Temporary MLflow store, synthetic evaluation.json."""

import json
import re

import pytest

from pm25_forecast.experiments import report, tracking

pytest.importorskip("mlflow")
pytest.importorskip("matplotlib")

SHA = "b" * 64
STATIONS = ["s1", "s2"]
HOURS = [f"{h:02d}" for h in range(24)]


def evaluation(mae, diff, ci, fragile=False):
    """The slice of the runner's evaluation.json that the report reads (+24 h only)."""
    cell = {"model": mae, "persistence": 4.3, "smoother": 3.9, "n_pairs": 10}
    return {
        "invalid_reasons": [],
        "plan": {"fold_starts": [f"2026-09-{d:02d}T00:00:00+00:00" for d in (1, 8, 15)]},
        "scores": {
            "24": {
                "baselines": {"persistence": {"mae": 4.3}, "smoother": {"mae": 3.9}},
                "by_fold": {str(i): {"diff_vs_smoother": diff + 0.01 * i} for i in range(3)},
                "mae_by_station": {s: dict(cell) for s in STATIONS},
                "mae_by_hour_utc": {h: dict(cell) for h in HOURS},
                "vs_smoother": {
                    "diff": diff,
                    "ci": ci,
                    "ci_sensitivity": {"24": ci, "168": ci},
                    "fragile": fragile,
                },
            }
        },
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    import mlflow

    for name in (tracking.STORE_ENV, tracking.ARTIFACT_ENV):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "db").mkdir()
    uri = f"sqlite:///{tmp_path / 'db' / 'm.db'}"
    monkeypatch.setenv(tracking.STORE_ENV, uri)
    mlflow.set_tracking_uri(uri)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tracking, "git_state", lambda: ("d" * 40, False))
    return tmp_path


def add_run(tmp, name, mae, diff, ci, *, ticket="KAN-90", fragile=False, ci_level=0.95, **tag_over):
    path = tmp / f"{name}.evaluation.json"
    path.write_text(json.dumps(evaluation(mae, diff, ci, fragile)))
    # the artifact is logged under its own file name; evaluation.json is the contract
    final = tmp / name / "evaluation.json"
    final.parent.mkdir(exist_ok=True)
    final.write_text(path.read_text())
    tags = {
        "ticket": ticket,
        "agent": "t",
        "model_name": name,
        "model_family": "f",
        "feature_set": "fs",
        "snapshot_sha256": SHA,
        "split": "dev",
        "prereg": f"p-{name}",
    }
    tags.update(tag_over)
    return tracking.log_run(
        f"kan78/{name}",
        f"{name}-run",
        params={"ci_level": ci_level},
        metrics={
            "mae_24h": mae,
            "rmse_24h": mae + 1,
            "skill_vs_persistence_24h": 1 - mae / 4.3,
            "n_pairs_24h": 10,
        },
        tags=tags,
        artifacts=[final],
    )


@pytest.fixture
def populated(store):
    add_run(store, "e90-a", 3.70, -0.20, [-0.40, -0.01])  # clean win on paper
    add_run(store, "e90-b", 3.80, -0.10, [-0.20, 0.05])
    add_run(store, "e90-c", 3.90, 0.0, None)  # the smoother itself
    add_run(store, "e90-abl-no-lags", 4.00, 0.1, [0.0, 0.2])
    add_run(store, "e90-a-bonferroni", 3.75, -0.2, [-0.5, 0.1], ci_level=0.998)
    add_run(store, "e91-x", 3.0, -0.8, [-1.0, -0.6], ticket="KAN-91")
    return store


def docs(tmp):
    d = tmp / "docs"
    (d / "prereg").mkdir(parents=True, exist_ok=True)
    (d / "e90-test.md").write_text(
        "# E90 test <b>experiment</b> (KAN-90)\n\nVerdict: **NO WIN** here.\n\n"
        "| a | b |\n|---|---|\n| 1 | `x<y` |\n\n- one\n- two with **bold**\n\n"
        "<script>alert(1)</script>\n"
    )
    (d / "prereg" / "p-e90-a.yaml").write_text(
        "hypothesis: Model A is better than the smoother at 24 h.\n"
        "decision_rule: CI vs smoother entirely below zero.\n"
    )
    return d


def test_collect_variants_kinds_and_order(populated):
    vs = report.collect_variants("KAN-90")
    assert [v.name for v in vs] == [
        "e90-a",
        "e90-a-bonferroni",
        "e90-b",
        "e90-c",
        "e90-abl-no-lags",
    ]
    kinds = {v.name: v.kind for v in vs}
    assert kinds["e90-a-bonferroni"] == "sensitivity"
    assert kinds["e90-abl-no-lags"] == "ablation"
    assert kinds["e90-a"] == "primary"
    by = {v.name: v for v in vs}
    assert by["e90-a"].status == "clean" and by["e90-b"].status == "none"
    assert by["e90-c"].status == "reference"
    assert by["e90-a"].baselines == {"persistence": 4.3, "smoother": 3.9}


def test_holdout_runs_are_never_read(populated):
    import mlflow

    client = tracking._client()
    exp = client.get_experiment_by_name("kan78/e90-a")
    (run,) = client.search_runs([exp.experiment_id])
    client.set_tag(run.info.run_id, "split", "holdout")
    mlflow.set_tracking_uri(tracking.tracking_uri())
    assert "e90-a" not in [v.name for v in report.collect_variants("KAN-90")]


def test_report_is_self_contained_with_all_plots(populated):
    out = populated / "out"
    path = report.write_ticket_report("KAN-90", out, docs(populated))
    text = path.read_text()
    assert text.count("data:image/png;base64,") == 5  # mae, paired, folds, breakdown, ablation
    assert 'src="http' not in text and 'href="http' not in text
    assert "<script" not in text.replace("&lt;script&gt;", "")
    assert "How to read this" in text and "entirely below zero" in text
    # prereg hypothesis and decision rule, narrative, verdict, status
    assert "Model A is better than the smoother" in text
    assert "NO WIN" in text and "written up" in text
    assert "&lt;b&gt;experiment&lt;/b&gt;" in text  # markdown input is escaped
    assert "<code>x&lt;y</code>" in text
    side = json.loads((out / "KAN-90.json").read_text())
    assert side["verdict"] == "NO WIN" and side["n_variants"] == 3
    assert "e90-a-bonferroni" in text  # listed in the table, flagged as sensitivity
    assert report.write_ticket_report("KAN-90", out, docs(populated)).stat().st_size < 1_500_000


def test_win_is_hedged_and_fragile_is_not_a_win(store):
    add_run(store, "e90-a", 3.70, -0.20, [-0.40, -0.01], fragile=True)
    add_run(store, "e90-b", 3.80, -0.10, [-0.20, 0.05])
    vs = report.collect_variants("KAN-90")
    assert "clearly beats" in report.plot_paired(vs).title
    assert "No variant clearly beats" in report.headline(vs)
    assert "unadjusted" in report.plot_paired(vs).caption


def test_clean_ci_headline_says_exploratory(populated):
    text = report.headline(report.collect_variants("KAN-90"))
    assert "1 of 2" in text and "unadjusted" in text and "not evidence of holdout" in text


def test_no_ci_below_zero_says_no_variant_beats(store):
    add_run(store, "e90-a", 3.70, -0.01, [-0.40, 0.2])
    vs = report.collect_variants("KAN-90")
    assert report.headline(vs).startswith("No variant beats the smoother")


def test_missing_evaluation_artifact_is_tolerated(store):
    tracking.log_run(
        "kan78/e90-n",
        "r",
        {},
        {"mae_24h": 3.0, "rmse_24h": 4.0},
        {
            "ticket": "KAN-90",
            "agent": "t",
            "model_name": "m",
            "model_family": "f",
            "feature_set": "f",
            "snapshot_sha256": SHA,
            "split": "dev",
            "prereg": "p",
        },
    )
    (v,) = report.collect_variants("KAN-90")
    assert not v.has_eval
    text = report.build_report("KAN-90", [v], report.Narrative(), [])[0]
    assert "No evaluation.json" in text and "No variant has an evaluation.json" in text


def test_unknown_ticket_and_bad_ticket(store, tmp_path):
    with pytest.raises(tracking.TrackingError, match="no valid dev runs"):
        report.write_ticket_report("KAN-1", tmp_path / "o")
    with pytest.raises(tracking.TrackingError, match="KAN-<n>"):
        report.write_ticket_report("x", tmp_path / "o")


def test_index_has_leaderboard_ticket_rows_and_warnings(populated, monkeypatch):
    out = populated / "out"
    report.write_ticket_report("KAN-90", out, docs(populated))
    other = "c" * 64
    add_run(populated, "e00-demo", 1.0, -0.1, [-0.2, 0.0], ticket="KAN-81", snapshot_sha256=other)
    monkeypatch.setattr(tracking, "git_state", lambda: ("e" * 40, True))  # the next run is dirty
    add_run(populated, "e92-dirty", 3.5, -0.3, [-0.5, 0.1], ticket="KAN-92")
    assert report.main(["index", "--out", str(out)]) == 0
    text = (out / "index.html").read_text()
    assert '<a href="KAN-90.html">KAN-90</a>' in text
    assert "NO WIN" in text and "written up" in text
    assert "no report yet" in text and "--ticket KAN-91" in text
    assert "DIRTY" in text
    assert "different snapshot_sha256 values" in text
    assert "kan78/e90-a" in text and text.count("<tr>") > 5


def test_cli_writes_report(populated, capsys):
    out = populated / "cli"
    d = docs(populated)
    assert report.main(["--ticket", "KAN-90", "--out", str(out), "--docs-dir", str(d)]) == 0
    assert (out / "KAN-90.html").is_file()
    assert "KB" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        report.main([])


def test_markdown_subset():
    html = report.markdown_to_html(
        "## Title\n\ntext with `c` and [l](https://x.y) and [bad](javascript:alert(1))\n\n"
        "1. a\n2. b\n\n```\ncode <x>\n```\n"
    )
    assert "<h3>Title</h3>" in html and "<ol><li>a</li><li>b</li></ol>" in html
    assert 'href="https://x.y"' in html and "javascript:" in html and 'href="javascript' not in html
    assert "<pre><code>code &lt;x&gt;</code></pre>" in html
    assert re.search(r"<p>text with <code>c</code>", html)


def test_verdict_only_from_explicit_line():
    assert report._verdict("# T\n\n**Caveat** here\nVerdict: **NO WIN** x") == "NO WIN"
    assert report._verdict("# T\n\n**Caveat** only\n") == ""


def test_verdict_claims_win():
    assert report.verdict_claims_win("WIN")
    assert not report.verdict_claims_win("NO WIN")
    assert not report.verdict_claims_win("PIVOT")


def test_index_flags_mismatch_and_missing_verdict():
    base = {"finding": "f", "outcome": "none"}
    assert "MISMATCH" in report._verdict_cell({**base, "verdict": "WIN"})
    assert "MISMATCH" in report._verdict_cell({**base, "outcome": "win", "verdict": "NO WIN"})
    ok = report._verdict_cell({**base, "verdict": "NO WIN"})
    assert "MISMATCH" not in ok and "Computed: f" in ok
    assert "no explicit verdict" in report._verdict_cell({**base, "verdict": ""})


def test_plot_mae_without_baselines_has_clean_caption(store):
    add_run(store, "e90-a", 3.70, -0.20, [-0.40, -0.01])
    (v,) = report.collect_variants("KAN-90")
    v.baselines = {}
    plot = report.plot_mae([v])
    assert plot.caption.startswith("Variants are sorted") and "None" not in plot.title
