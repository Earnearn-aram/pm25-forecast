from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast.backtest import Observation, Vintage
from pm25_forecast.experiments.kan87_coverage import coverage
from pm25_forecast.experiments.kan87_models import ridge_cams, row_features
from pm25_forecast.features.cams import CAMS_FEATURE_NAMES, build_cams_features, prepare_cams
from pm25_forecast.features.library_v2 import prepare_series

H = timedelta(hours=1)
T = datetime(2026, 11, 3, 12, tzinfo=UTC)
TARGET = T + 24 * H


def obs(ks, value=20.0, lag=H):
    return [Observation("a", T - k * H, value, T - k * H + lag) for k in ks]


def vin(valid, value, issue, avail=None, model="cams"):
    return Vintage("a", issue, valid, value, avail or issue + timedelta(hours=5), model)


def feats(o, v):
    return build_cams_features(prepare_series({"a": o}), prepare_cams({"a": v}), "a", T, TARGET)


def test_names_and_latest_vintage_by_issue_time():
    v = [vin(TARGET, 30.0, T - 48 * H), vin(TARGET, 40.0, T - 12 * H)]
    f = feats(obs(range(1, 4)), v)
    assert tuple(f) == CAMS_FEATURE_NAMES
    assert f["cams_pm25_target"] == 40.0
    assert f["cams_minus_latest"] == 20.0


def test_vintage_not_visible_before_issue_or_available_at():
    late_issue = vin(TARGET, 99.0, T + H)  # issued after the origin
    late_avail = vin(TARGET, 98.0, T - 6 * H, avail=T + H)  # issued, but not yet available
    ok = vin(TARGET, 10.0, T - 30 * H)
    f = feats(obs(range(1, 4)), [late_issue, late_avail, ok])
    assert f["cams_pm25_target"] == 10.0
    assert feats(obs(range(1, 4)), [late_issue, late_avail])["cams_pm25_target"] is None


def test_other_models_and_missing_available_at_ignored():
    bad = [vin(TARGET, 5.0, T - 6 * H, model="ecmwf"), vin(TARGET, 6.0, T - 6 * H)]
    bad[1] = Vintage("a", T - 6 * H, TARGET, 6.0, None, "cams")  # type: ignore[arg-type]
    assert feats(obs(range(1, 4)), bad)["cams_pm25_target"] is None


def test_recent_error_uses_only_matured_targets_and_real_forecasts():
    # CAMS says 10 for each of the last 30 hours (issued 40 h before each hour); actual is 20.
    v = [vin(T - k * H, 10.0, T - k * H - 40 * H) for k in range(1, 31)]
    f = feats(obs(range(1, 31)), v)
    assert f["cams_err_24h"] == pytest.approx(10.0)
    assert f["cams_err_72h"] == pytest.approx(10.0)
    # observation hour 1 not yet available at T (lag 2 h): dropped, others unaffected
    slow = obs(range(2, 31)) + obs([1], lag=2 * H)
    assert feats(slow, v)["cams_err_24h"] == pytest.approx(10.0)
    # a vintage issued after the hour it forecasts is a hindcast: not used for the error
    hind = [vin(T - k * H, 10.0, T - k * H + H, avail=T - 1 * H) for k in range(1, 31)]
    assert feats(obs(range(1, 31)), hind)["cams_err_24h"] is None


def test_error_needs_min_hours_and_nothing_imputed():
    v = [vin(T - k * H, 10.0, T - k * H - 40 * H) for k in range(1, 4)]
    assert feats(obs(range(1, 31)), v)["cams_err_24h"] is None
    assert feats(obs(range(1, 31)), [])["cams_minus_latest"] is None


def test_origin_must_be_on_the_hour():
    with pytest.raises(ValueError):
        build_cams_features(
            prepare_series({}), prepare_cams({}), "a", T + timedelta(minutes=5), TARGET
        )


def test_matched_rows_rule_same_with_and_without_cams():
    o = obs(range(1, 80))
    v = [vin(T - k * H, 10.0, T - k * H - 40 * H) for k in range(1, 31)]
    v.append(vin(TARGET, 30.0, T - 6 * H))
    prep, cams = prepare_series({"a": o}), prepare_cams({"a": v})
    with_c = row_features(prep, cams, "a", T, TARGET, with_cams=True)
    without = row_features(prep, cams, "a", T, TARGET, with_cams=False)
    assert with_c is not None and without is not None
    assert set(with_c) - set(without) == set(CAMS_FEATURE_NAMES)
    empty = prepare_cams({})
    assert row_features(prep, empty, "a", T, TARGET, with_cams=False) is None
    assert row_features(prep, empty, "a", T, TARGET, with_cams=True) is None


def test_model_without_vintages_forecasts_nothing():
    from pm25_forecast.experiments.evaluate import TrainSnapshot

    train = TrainSnapshot(T, ("a",), (24,), tuple(obs(range(0, 500))), ())
    predict = ridge_cams(train)

    class Snap:
        station_id = "a"
        prediction_time = T
        observations = tuple(obs(range(1, 80)))
        vintages = ()

    assert predict(Snap(), 24) is None  # type: ignore[arg-type]


def test_coverage_zero_without_vintages_and_counts_with_them():
    start = datetime(2026, 7, 31, 17, tzinfo=UTC)
    end = datetime(2026, 10, 26, tzinfo=UTC)
    n = int((end - start) / H)
    o = [Observation("a", start + i * H, 20.0, start + (i + 1) * H) for i in range(n)]
    assert coverage(o, [])["share_all_features"] == 0.0
    # a vintage inside the dev window is counted; one after H0 is never read
    h0 = datetime(2026, 10, 5, tzinfo=UTC)
    inside = Vintage("a", h0 - 4 * 24 * H, h0 - 2 * 24 * H, 30.0, h0 - 4 * 24 * H + 5 * H)
    after = Vintage("a", h0 + 24 * H, h0 + 48 * H, 30.0, h0 + 29 * H)
    res = coverage(o, [inside, after])
    assert res["vintages_read"] == 1
    assert res["with_target_vintage"] == 1
    assert res["with_all_features"] == 0  # the error features need >= 6 matched hours


def _synthetic_run(factory, params=None):
    import numpy as np

    from pm25_forecast.experiments.evaluate import run_evaluation

    rng = np.random.default_rng(7)
    start = datetime(2026, 7, 31, 17, tzinfo=UTC)
    end = datetime(2026, 10, 26, tzinfo=UTC)
    hours = [start + i * H for i in range(int((end - start) / H))]
    truth = [30 + 10 * np.sin(i / 12.0) + rng.normal(0, 2) for i in range(len(hours))]
    o = [Observation("a", t, float(v), t + H) for t, v in zip(hours, truth, strict=True)]
    # CAMS vintages for every valid hour from 2026-09-01: issued 36 h ahead, stored 5 h later
    v = [
        Vintage("a", t - 36 * H, t, float(x + rng.normal(0, 3)), t - 31 * H)
        for t, x in zip(hours, truth, strict=True)
        if t >= datetime(2026, 9, 1, tzinfo=UTC)
    ]
    return run_evaluation(o, v, fit=factory, horizons=[24], params=params)


@pytest.mark.parametrize(
    "family,params",
    [("ridge", None), ("lgbm", {"n_seeds": 1, "n_estimators": 20, "min_child_samples": 20})],
)
def test_models_run_through_the_runner_and_base_cams_use_same_rows(family, params):
    if family == "lgbm":
        # LightGBM's scikit-learn API needs scikit-learn, which lives in the experiments group only.
        pytest.importorskip("sklearn")
    from pm25_forecast.experiments import kan87_models as m

    cams = _synthetic_run(getattr(m, f"{family}_cams"), params)
    base = _synthetic_run(getattr(m, f"{family}_base"), params)
    n_cams, n_base = cams["scores"]["24"]["n_pairs"], base["scores"]["24"]["n_pairs"]
    assert n_cams > 0  # forecasts are produced once vintages exist
    assert n_cams == n_base  # identical rows with and without CAMS


def test_equal_issue_time_ties_break_on_available_at_not_order():
    a = Vintage("a", T - 12 * H, TARGET, 1.0, T - 8 * H)
    b = Vintage("a", T - 12 * H, TARGET, 2.0, T - 7 * H)  # same issue, stored later: wins
    for order in ([a, b], [b, a]):
        assert feats(obs(range(1, 4)), order)["cams_pm25_target"] == 2.0
