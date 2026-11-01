"""KAN-79: fold/holdout date arithmetic and minimum sample sizes (5 stations)."""

from datetime import UTC, datetime

import pytest

from pm25_forecast.experiments import folds as f

STATIONS = 5


def test_current_snapshot():
    r = f.derive_folds(datetime(2026, 10, 25, 23, tzinfo=UTC))
    assert r.end == datetime(2026, 10, 26, tzinfo=UTC)
    assert r.holdout_start == datetime(2026, 10, 5, tzinfo=UTC)
    assert r.holdout_folds == tuple(datetime(2026, 10, d, tzinfo=UTC) for d in (5, 12, 19))
    assert r.holdout_start - f.S0 >= f.MIN_DEV
    assert all(s >= f.S0 + f.INITIAL_TRAIN for s in r.dev_folds)
    assert len(r.dev_folds) >= 3
    assert r.dev_folds[-1] + f.FOLD == r.holdout_start


def test_target_cut_applies_even_with_later_data():
    r = f.derive_folds(datetime(2026, 11, 20, 5, tzinfo=UTC))
    assert r.end == f.TARGET_CUT


def test_partial_last_day_is_floored():
    r = f.derive_folds(datetime(2026, 10, 20, 12, tzinfo=UTC))
    assert r.end == datetime(2026, 10, 20, tzinfo=UTC)


def test_short_snapshot_has_no_holdout():
    r = f.derive_folds(datetime(2026, 10, 8, 23, tzinfo=UTC))  # D = 48 d 7 h < 49 d
    assert r.holdout_start is None
    assert r.holdout_folds == ()
    assert r.dev_folds  # dev-only folds still exist


def test_boundary_70_days_has_holdout():
    r = f.derive_folds(datetime(2026, 10, 9, 23, tzinfo=UTC))  # E = 10-10, D = 49 d + 7 h
    assert r.holdout_start is not None
    assert r.holdout_start - f.S0 >= f.MIN_DEV


def test_naive_rejected():
    with pytest.raises(ValueError):
        f.derive_folds(datetime(2026, 10, 1))  # noqa: DTZ001 - naive on purpose


def test_minimums_are_attainable_with_five_stations():
    assert f.HOLDOUT.days * 24 * STATIONS >= f.MIN_HOLDOUT_PAIRS  # 2,520 >= 2,100
    assert f.FOLD.days * 24 * STATIONS >= f.MIN_DEV_FOLD_PAIRS  # 840 >= 600
    assert f.HOLDOUT.days * 24 >= f.MIN_HOLDOUT_ORIGINS
    assert f.HOLDOUT.days * 24 * STATIONS > f.MIN_HOLDOUT_PAIRS
    assert f.HOLDOUT.days * 24 >= f.MIN_STATION_PAIRS
