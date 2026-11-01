"""KAN-82: walk-forward evaluation runner on synthetic data. No network, no database."""

import json
import math
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pm25_forecast.backtest import Observation
from pm25_forecast.experiments import evaluate as ev
from pm25_forecast.experiments import folds, tracking
from pm25_forecast.experiments.dataset import write_snapshot
from pm25_forecast.experiments.prereg import Prereg

HOUR = timedelta(hours=1)
LAST = datetime(2026, 10, 25, 23, tzinfo=UTC)
STATIONS = ("s1", "s2", "s3", "s4", "s5")


def make_obs(end=LAST, stations=STATIONS, seed=0, skip=()):
    rng = np.random.default_rng(seed)
    out = []
    for s_i, s in enumerate(stations):
        noise = 0.0
        ts = folds.S0
        while ts <= end:
            noise = 0.9 * noise + rng.normal(0, 3)
            hour = ts.hour
            value = max(0.0, 25 + 8 * math.sin(2 * math.pi * hour / 24) + 2 * s_i + noise)
            if (s, ts) not in skip:
                out.append(Observation(s, ts, round(value, 3), ts + HOUR))
            ts += HOUR
    return out


@pytest.fixture(scope="module")
def obs():
    return make_obs()


def prereg(**over):
    base = {
        "id": "e99-test-v1",
        "ticket": "KAN-99",
        "hypothesis": "A test model lowers MAE.",
        "model_family": "test",
        "model": "test-model",
        "feature_set": "none",
        "params": {"k": 1},
        "horizons": [24],
        "expected_effect": {
            "metric": "diff_vs_smoother_24h",
            "direction": "negative",
            "magnitude": -0.2,
        },
        "decision_rule": "Promote only if the holdout gate passes.",
        "split": "dev",
        "final": False,
        "author_agent": "test-agent",
    }
    base.update(over)
    return Prereg.model_validate(base)


FINAL = {"split": "holdout", "final": True}


# ---- baselines -------------------------------------------------------------------------


def lookup_of(rows):
    table = {o.ts: o for o in rows}
    return table.get


def series(values, start=datetime(2026, 9, 1, tzinfo=UTC)):
    return [
        Observation("s", start + i * HOUR, v, start + (i + 1) * HOUR) for i, v in enumerate(values)
    ]


def test_persistence_uses_newest_reading_within_age_limit():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    rows = series([10.0, 20.0, 30.0, 40.0])  # ts t0 .. t0+3h
    origin = t0 + 4 * HOUR  # newest allowed reading is t0+3h
    assert ev.persistence_value(lookup_of(rows), origin) == 40.0
    assert ev.persistence_value(lookup_of(rows[:1]), origin) is None  # 4 h old
    # ts == origin - 3h is still allowed
    assert ev.persistence_value(lookup_of(rows[1:2]), origin) == 20.0


def test_persistence_ignores_origin_hour_and_unavailable_rows():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    rows = series([10.0, 20.0, 30.0, 40.0, 99.0])  # last row has ts == origin
    assert ev.persistence_value(lookup_of(rows), t0 + 4 * HOUR) == 40.0
    late = [Observation("s", t0 + 3 * HOUR, 40.0, t0 + 9 * HOUR)]
    assert ev.persistence_value(lookup_of(late), t0 + 4 * HOUR) is None


def test_smoother_blends_latest_and_mean_and_falls_back():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    rows = series([10.0] * 23 + [20.0])  # hours t0 .. t0+23, origin t0+24
    origin = t0 + 24 * HOUR
    mean24 = (10 * 23 + 20) / 24
    assert ev.smoother_value(lookup_of(rows), origin) == pytest.approx(20 + 0.5 * (mean24 - 20))
    only_latest = rows[-1:]
    # one reading: mean24 == latest, so the smoother equals persistence
    assert ev.smoother_value(lookup_of(only_latest), origin) == 20.0
    assert ev.smoother_value(lookup_of([]), origin) is None


def test_seasonal_naive_never_uses_the_origin_hour():
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    rows = series(list(range(100)))
    origin = t0 + 60 * HOUR
    # +6 h: target t0+66h; previous day t0+42h <= origin-1h
    assert ev.seasonal_naive_value(lookup_of(rows), origin, 6) == 42
    # +24 h: target - 24h == origin (not allowed) -> target - 48h == origin - 24h = t0+36h
    assert ev.seasonal_naive_value(lookup_of(rows), origin, 24) == 36
    assert ev.seasonal_naive_value(lookup_of(rows[:30]), origin, 24) is None


# ---- plan and cohort -------------------------------------------------------------------


def test_plan_matches_protocol_folds(obs):
    dev = ev.make_plan(LAST, "dev")
    hold = ev.make_plan(LAST, "holdout")
    assert hold.h0 == datetime(2026, 10, 5, tzinfo=UTC)
    assert hold.end == datetime(2026, 10, 26, tzinfo=UTC)
    assert [s.day for s, _ in hold.folds] == [5, 12, 19]
    assert hold.folds[-1][1] == hold.end
    assert dev.limit == hold.h0
    assert len(dev.folds) == 5  # weekly folds back to S0 + 28 d
    assert all(a < b for a, b in dev.folds) and dev.folds[-1][1] == hold.h0


def test_holdout_refused_for_a_short_snapshot():
    with pytest.raises(ev.EvaluationError, match="too short"):
        ev.make_plan(datetime(2026, 9, 30, 23, tzinfo=UTC), "holdout")
    assert ev.make_plan(datetime(2026, 9, 30, 23, tzinfo=UTC), "dev").h0 is None


def test_cohort_needs_80_percent_coverage():
    start, end = folds.S0, folds.S0 + timedelta(hours=100)
    rows = [Observation("full", start + i * HOUR, 1.0, start) for i in range(100)]
    rows += [Observation("good", start + i * HOUR, 1.0, start) for i in range(80)]
    rows += [Observation("thin", start + i * HOUR, 1.0, start) for i in range(79)]
    assert ev.select_cohort(rows, start, end) == ["full", "good"]


# ---- the run ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def persistence_run(obs):
    return ev.run_evaluation(obs, [], fit=ev.persistence, horizons=(6, 12, 24))


def test_persistence_dev_run_shape(persistence_run):
    r = persistence_run
    assert r["split"] == "dev" and r["plan"]["n_folds"] == 5
    assert r["plan"]["H0"].startswith("2026-10-05")
    s = r["scores"]["24"]
    assert s["n_pairs"] == 5 * 7 * 24 * len(STATIONS) == s["eligible_pairs"]
    assert s["coverage"] == 1.0
    assert s["mae"] == pytest.approx(s["baselines"]["persistence"]["mae"])
    assert s["skill_vs_persistence"] == 0
    assert s["vs_persistence"]["diff"] == 0 and "ci" not in s["vs_persistence"]
    assert "ci" in s["vs_smoother"] and s["vs_smoother"]["dm"]["167"] is not None
    assert r["invalid_reasons"] == []
    assert 0 <= s["spike"]["recall"] <= 1
    assert len(s["mae_by_hour_utc"]) == 24 and set(s["mae_by_station"]) == set(STATIONS)


def test_smoother_run_beats_persistence_on_smooth_noise(obs):
    r = ev.run_evaluation(obs, [], fit=ev.smoother, horizons=(24,))
    s = r["scores"]["24"]
    assert s["vs_smoother"]["diff"] == 0 and "ci" not in s["vs_smoother"]
    lo, hi = s["vs_persistence"]["ci"]
    assert hi < 0 and lo < hi  # AR noise: the 24h mean is a better +24h guess than the last hour
    assert s["skill_vs_persistence"] > 0
    assert s["baselines"]["smoother"]["mae"] == pytest.approx(s["mae"])


def test_seasonal_naive_run_scores_only_rows_with_a_forecast(obs):
    r = ev.run_evaluation(obs, [], fit=ev.seasonal_naive, horizons=(24,))
    s = r["scores"]["24"]
    assert s["n_pairs"] == s["eligible_pairs"]  # no gaps in the synthetic series
    assert s["mae"] == pytest.approx(s["baselines"]["seasonal_naive"]["mae"])


def test_dev_run_never_reads_holdout_labels(obs):
    h0 = ev.make_plan(LAST, "dev").limit
    tampered = [
        Observation(o.station_id, o.ts, 500.0, o.available_at) if o.ts >= h0 else o for o in obs
    ]
    a = ev.run_evaluation(obs, [], fit=ev.smoother, horizons=(24,))
    b = ev.run_evaluation(tampered, [], fit=ev.smoother, horizons=(24,))
    assert a["scores"] == b["scores"]


def test_targets_from_the_kan74_cut_are_excluded(obs):
    extra = make_obs(end=LAST + timedelta(days=3))
    r = ev.run_evaluation(extra, [], fit=ev.persistence, horizons=(24,))
    assert r["plan"]["E"] == ev.make_plan(LAST, "holdout").end.isoformat()


def test_no_leakage_into_fit_or_predict(obs):
    seen = {"fits": [], "calls": 0}

    def spy(train):
        assert max(o.ts for o in train.observations) <= train.refit_time
        assert max(o.available_at for o in train.observations) <= train.refit_time
        seen["fits"].append(train.refit_time)
        return lambda snap, horizon: _check(snap, horizon)

    def _check(snap, horizon):
        seen["calls"] += 1
        assert all(o.ts <= snap.prediction_time - HOUR for o in snap.observations)
        assert all(o.available_at <= snap.prediction_time for o in snap.observations)
        assert snap.station_id in STATIONS
        return 20.0

    ev.run_evaluation(obs, [], fit=spy, horizons=(24,))
    plan = ev.make_plan(LAST, "dev")
    # one refit per week for the live model (+2 fresh instances per fold for the lag probe)
    assert seen["fits"][0::3] == [start - 24 * HOUR for start, _ in plan.folds]
    assert seen["calls"] > 1000


def test_refit_offset_uses_largest_horizon(obs):
    refits = []

    def spy(train):
        refits.append(train.refit_time)
        return lambda snap, horizon: 20.0

    ev.run_evaluation(obs, [], fit=spy, horizons=(6, 24))
    assert refits[0] == ev.make_plan(LAST, "dev").folds[0][0] - 24 * HOUR


def test_candidate_missing_forecasts_invalidate_coverage(obs):
    def sparse(train):
        return lambda snap, horizon: None if snap.prediction_time.hour % 2 else 20.0

    r = ev.run_evaluation(obs, [], fit=sparse, horizons=(24,))
    assert r["scores"]["24"]["coverage"] < 0.6
    assert any("covers" in x for x in r["invalid_reasons"])


def test_negative_clipped_and_nonfinite_rejected(obs):
    r = ev.run_evaluation(obs, [], fit=lambda t: lambda s, h: -5.0, horizons=(24,))
    assert r["counters"]["clipped"] > 0 and r["scores"]["24"]["mae"] > 0
    with pytest.raises(ev.EvaluationError, match="non-finite"):
        ev.run_evaluation(obs, [], fit=lambda t: lambda s, h: math.nan, horizons=(24,))


def test_spike_metrics_need_50_observed_spikes():
    spiky = make_obs(seed=1)
    spiky = [Observation(o.station_id, o.ts, o.value + 25, o.available_at) for o in spiky]
    r = ev.run_evaluation(spiky, [], fit=ev.persistence, horizons=(24,))
    sp = r["scores"]["24"]["spike"]
    assert sp["tp"] + sp["fn"] >= 50
    assert 0 <= sp["recall"] <= 1 and 0 <= sp["precision"] <= 1


def test_spike_counts_strictly_above_threshold():
    a = np.array([37.5, 37.6, 50.0] * 30)
    p = np.array([40.0, 37.5, 10.0] * 30)
    sp = ev._spike(a, p)
    assert (sp["tp"], sp["fn"], sp["fp"]) == (0, 60, 30)
    assert sp["recall"] == 0.0 and sp["precision"] == 0.0
    none = ev._spike(a, np.zeros_like(a))
    assert none["recall"] == 0.0 and none["precision"] is None  # nothing predicted: undefined
    assert ev._spike(a[:30], p[:30])["recall"] is None  # < 50 observed spikes


def test_metric_names_are_the_protocol_names(persistence_run):
    metrics = ev.metrics_from_result(persistence_run)
    assert set(metrics) <= tracking.ALLOWED_METRICS
    assert {"mae_24h", "rmse_24h", "n_pairs_24h", "dm_pvalue_vs_smoother_24h"} <= set(metrics)
    assert metrics["diff_vs_persistence_ci_low_24h"] is None  # degenerate: omitted, never 0


def test_ci_level_is_bonferroni_in_the_number_of_finalists():
    assert ev.ci_level_for(1) == 0.95
    assert ev.ci_level_for(2) == pytest.approx(0.975)
    with pytest.raises(ev.EvaluationError):
        ev.ci_level_for(0)


def test_dev_run_of_persistence_is_fast():
    # Runtime guard of the ticket: a full dev run of persistence on ~85 days of 5 stations.
    data = make_obs(seed=5)
    started = time.monotonic()
    ev.run_evaluation(data, [], fit=ev.persistence)
    assert time.monotonic() - started < 60  # 300 s is the budget on a laptop; this is 5x tighter


# ---- holdout guards --------------------------------------------------------------------


@pytest.fixture
def clean_tracker(monkeypatch):
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))
    monkeypatch.setattr(ev, "holdout_already_scored", lambda _id: False)


def test_holdout_allowed_for_final_prereg(clean_tracker):
    ev.check_holdout_allowed(prereg(**FINAL), flag=True, logging_on=True)


@pytest.mark.parametrize(
    ("pre", "flag", "log", "match"),
    [
        (None, True, True, "final"),
        (prereg(), True, True, "final"),
        (prereg(split="holdout"), True, True, "final"),
        (prereg(**FINAL), False, True, "--holdout"),
        (prereg(**FINAL), True, False, "logged"),
    ],
)
def test_holdout_refused(clean_tracker, pre, flag, log, match):
    with pytest.raises(ev.EvaluationError, match=match):
        ev.check_holdout_allowed(pre, flag=flag, logging_on=log)


def test_holdout_refused_when_dirty_or_already_scored(monkeypatch):
    monkeypatch.setattr(ev, "holdout_already_scored", lambda _id: False)
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, True))
    with pytest.raises(ev.EvaluationError, match="dirty"):
        ev.check_holdout_allowed(prereg(**FINAL), flag=True, logging_on=True)
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))
    monkeypatch.setattr(ev, "holdout_already_scored", lambda _id: True)
    with pytest.raises(ev.EvaluationError, match="already scored"):
        ev.check_holdout_allowed(prereg(**FINAL), flag=True, logging_on=True)


def test_holdout_run_scores_three_weekly_folds(obs):
    r = ev.run_evaluation(obs, [], fit=ev.smoother, split="holdout", horizons=(24,), ci_level=0.975)
    s = r["scores"]["24"]
    assert r["plan"]["n_folds"] == 3 and set(s["by_fold"]) == {"0", "1", "2"}
    assert s["n_pairs"] == 3 * 7 * 24 * len(STATIONS)
    assert r["invalid_reasons"] == []


# ---- end to end: CLI, snapshot on disk, MLflow ------------------------------------------


@pytest.fixture
def snapshot_dir(tmp_path, obs):
    rows = [(o.station_id, o.ts, o.value, "air4thai", o.available_at, o.available_at) for o in obs]
    write_snapshot(tmp_path / "snap", rows, [], kan74_repaired=False, git_sha="x")
    return tmp_path / "snap"


@pytest.fixture
def store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    (tmp_path / "store").mkdir()
    uri = f"sqlite:///{tmp_path / 'store' / 'm.db'}"
    monkeypatch.setenv(tracking.STORE_ENV, uri)
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))
    return mlflow


def test_cli_dry_run_writes_artifact_only(snapshot_dir, tmp_path, capsys):
    code = ev.main(
        [
            *("--snapshot", str(snapshot_dir)),
            *("--model", "pm25_forecast.experiments.evaluate:smoother"),
            *("--no-log", "--out-dir", str(tmp_path / "out")),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "split=dev folds=5" in out and "+24h n=4200" in out
    saved = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert saved["snapshot_sha256"] == tracking.snapshot_hash(snapshot_dir)


def test_cli_logs_three_baselines_to_the_tracker(snapshot_dir, store):
    for name in ev.BASELINE_NAMES:
        spec = f"pm25_forecast.experiments.evaluate:{name}"
        assert ev.main(["--snapshot", str(snapshot_dir), "--model", spec]) == 0
    rows = tracking.collect_leaderboard(split="dev")
    assert {r["experiment"] for r in rows} == {
        f"kan78/e04-{n.replace('_', '-')}" for n in ev.BASELINE_NAMES
    }
    run = store.MlflowClient().search_runs(
        [store.MlflowClient().get_experiment_by_name("kan78/e04-smoother").experiment_id]
    )[0]
    assert run.data.tags["prereg"] == "exploratory" and run.data.tags["split"] == "dev"
    assert run.data.metrics["n_pairs_24h"] == 5 * 7 * 24 * len(STATIONS)
    assert float(run.data.params["ci_level"]) == 0.95 and run.data.params["bootstrap_B"] == "10000"


def test_cli_prereg_run_and_holdout_refusal(snapshot_dir, store, tmp_path, capsys):
    import yaml

    good = prereg(horizons=[24]).model_dump(mode="json")
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(good))
    spec = "pm25_forecast.experiments.evaluate:smoother"
    assert ev.main(["--prereg", str(path), "--snapshot", str(snapshot_dir), "--model", spec]) == 0
    client = store.MlflowClient()
    run = client.search_runs([client.get_experiment_by_name("kan78/e99-test-v1").experiment_id])[0]
    assert run.data.tags["prereg"] == "e99-test-v1" and run.data.tags["model_name"] == "test-model"
    # not final -> the holdout flag is refused and nothing new is logged
    args = ["--prereg", str(path), "--snapshot", str(snapshot_dir), "--model", spec, "--holdout"]
    assert ev.main(args) == 2
    assert not [r for r in client.search_runs(["1", "2"]) if r.data.tags.get("split") == "holdout"]


def test_final_holdout_logs_once_then_refuses_rerun(snapshot_dir, store, tmp_path):
    import yaml

    path = tmp_path / "final.yaml"
    path.write_text(yaml.safe_dump(prereg(horizons=[24], **FINAL).model_dump(mode="json")))
    args = [
        "--prereg", str(path), "--snapshot", str(snapshot_dir), "--holdout",
        "--model", "pm25_forecast.experiments.evaluate:smoother",
    ]  # fmt: skip
    assert ev.main(args) == 0
    assert tracking.collect_leaderboard(split="holdout")[0]["experiment"] == "kan78/e99-test-v1"
    assert ev.holdout_already_scored("e99-test-v1")
    assert ev.main(args) == 2


def test_factory_receives_frozen_params_when_it_asks(obs):
    got = {}

    def fit(train, params):
        got.update(params)
        return lambda snap, horizon: 20.0

    ev.run_evaluation(obs, [], fit=fit, horizons=(24,), params={"k": 7})
    assert got == {"k": 7}


def test_bad_model_spec():
    with pytest.raises(ev.EvaluationError):
        ev.resolve_factory("no_colon")
    with pytest.raises(ev.EvaluationError):
        ev.resolve_factory("pm25_forecast.experiments.evaluate:HOUR")


# ---- review fixes: holdout inputs come from the prereg only; fail closed -----------------


@pytest.mark.parametrize("extra", [{"horizons": [24]}, {"finalists": 1}])
def test_holdout_rejects_cli_horizons_and_finalists(snapshot_dir, extra):
    with pytest.raises(ev.EvaluationError, match="not allowed with --holdout"):
        ev.execute(
            snapshot=snapshot_dir,
            model="pm25_forecast.experiments.evaluate:smoother",
            prereg=prereg(**FINAL),
            holdout=True,
            **extra,
        )


def test_cli_holdout_with_horizons_exits_2(snapshot_dir, tmp_path):
    import yaml

    path = tmp_path / "f.yaml"
    path.write_text(yaml.safe_dump(prereg(horizons=[24], **FINAL).model_dump(mode="json")))
    args = ["--prereg", str(path), "--snapshot", str(snapshot_dir), "--holdout", "--horizons", "24"]
    assert ev.main([*args, "--model", "pm25_forecast.experiments.evaluate:smoother"]) == 2


def test_ci_level_comes_from_prereg_finalists(snapshot_dir, tmp_path):
    two = prereg(horizons=[24], finalists=2)
    res = ev.execute(
        snapshot=snapshot_dir,
        model="pm25_forecast.experiments.evaluate:smoother",
        prereg=two,
        log=False,
        out_dir=tmp_path,
    )
    assert res["ci_level"] == pytest.approx(0.975)


def test_holdout_check_failing_store_refuses(monkeypatch):
    monkeypatch.setattr(tracking, "git_state", lambda: ("c" * 40, False))

    def boom(_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(ev, "holdout_already_scored", boom)
    with pytest.raises(ev.EvaluationError, match="cannot check"):
        ev.check_holdout_allowed(prereg(**FINAL), flag=True, logging_on=True)


def test_holdout_already_scored_rejects_unsafe_id():
    with pytest.raises(ValueError, match="unsafe"):
        ev.holdout_already_scored("x' or tags.split = 'dev")


# ---- the documented example model ------------------------------------------------------


def test_ridge_example_runs_through_the_runner(obs):
    from pm25_forecast.experiments.examples import ridge_linear

    res = ev.run_evaluation(obs, [], fit=ridge_linear.fit, horizons=(6, 24), params={"alpha": 5})
    s = res["scores"]["24"]
    assert s["coverage"] == 1.0 and res["invalid_reasons"] == []
    assert s["n_pairs"] == 5 * 7 * 24 * len(STATIONS)
    assert s["vs_persistence"]["ci"][1] < 0  # on AR noise a linear blend beats persistence
    assert np.isfinite(s["mae"]) and "ci" in s["vs_smoother"]


def test_ridge_example_resolves_as_a_cli_model_and_takes_params(obs):
    fit, name = ev.resolve_factory("pm25_forecast.experiments.examples.ridge_linear:fit")
    assert name == "fit"
    train = ev.TrainSnapshot(folds.S0 + timedelta(days=40), STATIONS, (24,), tuple(obs[:2000]), ())
    assert callable(fit(train, params={"alpha": 1.0}))


# ---- KAN-96: gate 10.3 (per-station vs persistence) ----------------------------------------


def _persistence_plus(delta_for):
    def fit(train):
        def predict(snap, horizon):
            base = ev.persistence_value(ev._snapshot_lookup(snap), snap.prediction_time)
            return None if base is None else base + delta_for(snap.station_id)

        return predict

    return fit


def test_station_gate_passes_for_a_better_model_and_for_persistence_itself(obs):
    better = ev.run_evaluation(obs, [], fit=ev.smoother, horizons=(24,))
    gate = better["scores"]["24"]["station_gate"]
    assert gate["status"] == "pass" and gate["failing"] == []
    assert gate["worst_rel_diff"] < 0 and gate["worst_rel_diff_ci_high"] < 0.05
    assert set(gate["stations"]) == set(STATIONS)
    same = ev.run_evaluation(obs, [], fit=ev.persistence, horizons=(24,))["scores"]["24"]
    assert same["station_gate"]["status"] == "pass"
    assert same["station_gate"]["worst_rel_diff"] == 0


def test_station_gate_fails_when_one_station_is_clearly_worse(obs):
    fit = _persistence_plus(lambda s: 15.0 if s == "s3" else 0.0)
    r = ev.run_evaluation(obs, [], fit=fit, horizons=(24,))
    gate = r["scores"]["24"]["station_gate"]
    assert gate["status"] == "fail" and gate["failing"] == ["s3"]
    s3 = gate["stations"]["s3"]
    assert s3["rel_diff"] > 0.05 and s3["ci"][0] > 0.05 and not s3["ci_includes_limit"]
    assert gate["stations"]["s1"]["rel_diff"] == 0
    assert gate["worst_station"] == "s3"
    m = ev.metrics_from_result(r)
    assert m["stations_worse_than_limit_24h"] == 1
    assert m["station_worst_rel_diff_24h"] == pytest.approx(s3["rel_diff"])
    assert m["station_worst_rel_diff_ci_high_24h"] == pytest.approx(s3["ci"][1])
    assert set(m) <= tracking.ALLOWED_METRICS


def _gate_arrays(rel, n=300, grid=400):
    """One station, persistence abs error 10 everywhere, model abs error 10 * (1 + rel)."""
    stations = np.array(["a"] * n)
    origins = np.arange(n)
    return stations, origins, np.full(n, 10 * (1 + rel)), np.full(n, 10.0), grid


@pytest.mark.parametrize(("rel", "status"), [(0.04, "pass"), (0.05, "pass"), (0.06, "fail")])
def test_station_gate_threshold_is_strictly_more_than_five_percent(rel, status):
    gate = ev.station_gate(*_gate_arrays(rel), 0.95, ["a"])
    assert gate["status"] == status
    assert gate["stations"]["a"]["rel_diff"] == pytest.approx(rel)


def test_station_gate_flags_ci_containing_the_limit_without_failing_it():
    stations, origins, _, pers, grid = _gate_arrays(0.0)
    # worse by 50% in some 72 h blocks, better by 44% in others: pooled about +4.9%, very uncertain
    rel = np.where((origins // 72) % 2 == 0, 0.5, -0.44)
    model = pers * (1 + rel)
    gate = ev.station_gate(stations, origins, np.abs(model), pers, grid, 0.95, ["a"])
    e = gate["stations"]["a"]
    assert e["rel_diff"] < 0.05 and gate["status"] == "pass"
    assert e["ci"][0] < 0.05 < e["ci"][1] and e["ci_includes_limit"]
    assert gate["ci_includes_limit"] == ["a"]


def test_station_gate_is_insufficient_when_too_many_stations_lack_pairs():
    st, org, model, pers, grid = _gate_arrays(0.0, n=100)  # 100 < 250 pairs
    gate = ev.station_gate(st, org, model, pers, grid, 0.95, ["a", "b"])
    assert gate["below_min_pairs"] == ["a", "b"] and gate["status"] == "insufficient_data"
    assert "rel_diff" not in gate["stations"]["b"]  # no pairs: undefined, not zero
    assert gate["worst_station"] == "a"


# ---- KAN-96: gate 10.4 (serving-lag probe) -------------------------------------------------


def lagged_obs(lag_h=3):
    return [Observation(o.station_id, o.ts, o.value, o.ts + lag_h * HOUR) for o in make_obs()]


def newest_row_model(train):
    """Leaky: newest row it was handed, whatever its ts or availability."""

    def predict(snap, horizon):
        rows = [o for o in snap.observations if o.ts <= snap.prediction_time]
        return max(rows, key=lambda o: o.ts).value if rows else None

    return predict


def origin_hour_model(train):
    """Leaky: uses the reading of the origin hour itself when it is in the snapshot."""

    def predict(snap, horizon):
        by_ts = {o.ts: o for o in snap.observations}
        row = by_ts.get(snap.prediction_time) or by_ts.get(snap.prediction_time - HOUR)
        return None if row is None else row.value

    return predict


def future_vintage_model(train):
    def predict(snap, horizon):
        later = [v for v in snap.vintages if v.issue_time > snap.prediction_time]
        return later[0].value if later else 20.0

    return predict


def test_lag_probe_passes_for_the_builtin_models(obs):
    for fit in (ev.persistence, ev.smoother, ev.seasonal_naive):
        lag = ev.run_evaluation(obs, [], fit=fit, horizons=(24,))["serving_lag"]
        assert lag["status"] == "pass" and lag["mismatches"] == 0, fit
        assert lag["checked"] >= 5 * ev.LAG_PROBE_ORIGINS_PER_FOLD  # 5 folds x origins x stations


def test_lag_probe_passes_for_a_model_that_filters_by_availability():
    data = lagged_obs()

    def careful(train):
        def predict(snap, horizon):
            rows = [
                o
                for o in snap.observations
                if o.ts <= snap.prediction_time - HOUR and o.available_at <= snap.prediction_time
            ]
            return max(rows, key=lambda o: o.ts).value if rows else None

        return predict

    r = ev.run_evaluation(data, [], fit=careful, horizons=(24,))
    assert r["serving_lag"]["status"] == "pass" and r["invalid_reasons"] == []


@pytest.mark.parametrize("leaky", [newest_row_model, origin_hour_model])
def test_lag_probe_catches_a_model_that_uses_forbidden_rows(obs, leaky):
    r = ev.run_evaluation(obs, [], fit=leaky, horizons=(24,))
    lag = r["serving_lag"]
    assert lag["status"] == "fail" and lag["mismatches"] > 0
    assert (
        lag["examples"][0]["with_allowed_rows_only"]
        != lag["examples"][0]["with_forbidden_rows_added"]
    )
    assert any("serving-lag check failed" in x for x in r["invalid_reasons"])
    assert ev.metrics_from_result(r)["serving_lag_mismatches"] == lag["mismatches"]


def test_lag_probe_catches_use_of_rows_not_yet_available_at_the_origin():
    # Every reading arrives 3 h late: rows with ts in (origin-3h, origin-1h] ARE older than the
    # lag but NOT available. A model that only checks ts passes the snapshot filter by luck of
    # the runner, and must be caught by the probe.
    def ts_only(train):
        def predict(snap, horizon):
            rows = [o for o in snap.observations if o.ts <= snap.prediction_time - HOUR]
            return max(rows, key=lambda o: o.ts).value if rows else None

        return predict

    r = ev.run_evaluation(lagged_obs(), [], fit=ts_only, horizons=(24,))
    assert r["serving_lag"]["status"] == "fail"


def test_lag_probe_catches_vintages_issued_after_the_origin(obs):
    from pm25_forecast.backtest import Vintage

    vin = [
        Vintage(
            "s1",
            folds.S0 + i * 6 * HOUR,
            folds.S0 + i * 6 * HOUR + 24 * HOUR,
            30.0,
            folds.S0 + i * 6 * HOUR + HOUR,
            "cams",
        )
        for i in range(0, 400)
    ]
    r = ev.run_evaluation(obs, vin, fit=future_vintage_model, horizons=(24,))
    assert r["serving_lag"]["status"] == "fail"


def test_lag_probe_is_inconclusive_when_the_model_rejects_extra_rows(obs):
    def strict(train):
        def predict(snap, horizon):
            assert all(o.ts < snap.prediction_time for o in snap.observations)
            return 20.0

        return predict

    r = ev.run_evaluation(obs, [], fit=strict, horizons=(24,))
    lag = r["serving_lag"]
    assert lag["status"] == "inconclusive" and lag["errors"] > 0 and lag["mismatches"] == 0
    assert r["invalid_reasons"] == []


def test_lag_probe_treats_a_noisy_model_as_inconclusive_not_leaky(obs):
    rng = np.random.default_rng(7)

    def noisy(train):
        return lambda snap, horizon: 20.0 + float(rng.normal())

    r = ev.run_evaluation(obs, [], fit=noisy, horizons=(24,))
    lag = r["serving_lag"]
    assert lag["status"] == "inconclusive" and lag["mismatches"] == 0 and lag["errors"] > 0
    assert r["invalid_reasons"] == []


def test_lag_probe_never_touches_the_live_predictor(obs):
    """Fresh instances answer the probe: a cache keyed by (station, origin) cannot hide a leak,
    and poisoned rows cannot corrupt the live model's state."""
    live_seen = []

    def cached_leaky(train):
        cache = {}

        def predict(snap, horizon):
            key = (snap.station_id, snap.prediction_time, horizon)
            if key not in cache:
                cache[key] = max(snap.observations, key=lambda o: o.ts).value
            return cache[key]

        live_seen.append(cache)
        return predict

    r = ev.run_evaluation(obs, [], fit=cached_leaky, horizons=(24,))
    assert r["serving_lag"]["status"] == "fail"
    # the live instance (first of each fold's three) only ever saw allowed rows
    assert all(v <= 200 for cache in live_seen[0::3] for v in cache.values())


def test_lag_probe_runtime_with_a_nontrivial_model(obs):
    from pm25_forecast.experiments.examples import ridge_linear

    start = time.perf_counter()
    r = ev.run_evaluation(obs, [], fit=ridge_linear.fit, horizons=(6, 12, 24))
    assert r["serving_lag"]["status"] == "pass"
    assert time.perf_counter() - start < 240


def test_lag_probe_is_deterministic(obs):
    a = ev.run_evaluation(obs, [], fit=newest_row_model, horizons=(24,))["serving_lag"]
    b = ev.run_evaluation(obs, [], fit=newest_row_model, horizons=(24,))["serving_lag"]
    assert a == b


def test_gate_metrics_are_logged_and_written_to_the_artifact(snapshot_dir, store, tmp_path):
    out = tmp_path / "out"
    result = ev.execute(
        snapshot=snapshot_dir,
        model="pm25_forecast.experiments.evaluate:smoother",
        out_dir=out,
        horizons=[24],
    )
    saved = json.loads((out / "evaluation.json").read_text())
    assert saved["serving_lag"]["status"] == "pass"
    assert saved["scores"]["24"]["station_gate"]["status"] == "pass"
    run = tracking._client().get_run(result["run_id"])
    assert run.data.metrics["serving_lag_mismatches"] == 0
    assert run.data.metrics["serving_lag_checks"] > 0
    assert run.data.metrics["stations_worse_than_limit_24h"] == 0
    assert "station_worst_rel_diff_24h" in run.data.metrics


def test_tracker_accepts_the_gate_metrics_and_still_rejects_unknown_ones():
    ok = {"mae_24h": 1.0, "rmse_24h": 2.0, "serving_lag_checks": 10, "serving_lag_mismatches": 0}
    assert tracking.validate_metrics(ok)["serving_lag_mismatches"] == 0
    with pytest.raises(tracking.TrackingError, match="unknown"):
        tracking.validate_metrics({**ok, "serving_lag_checks_6h": 1})
