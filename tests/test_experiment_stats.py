"""KAN-82: bootstrap CI, Newey-West / Diebold-Mariano and Student-t, against known values."""

import math

import numpy as np
import pytest

from pm25_forecast.experiments import stats

# Two-sided p-values from scipy.stats.t.sf (reference run once; scipy is not a dependency).
T_CASES = [
    (2.0, 10, 0.07338803477074037),
    (2.0, 5, 0.10193947882985832),
    (1.0, 1, 0.5),
    (2.228138851986274, 10, 0.05),
    (3.4641016151377544, 2, 0.07417990022744855),
    (1.96, 1_000_000, 0.04999606758526983),
    (0.5, 30, 0.6207230048851272),
    (5.0, 1500, 6.40710056889011e-07),
]


@pytest.mark.parametrize(("t", "df", "p"), T_CASES)
def test_student_t_two_sided_p(t, df, p):
    assert stats.student_t_two_sided_p(t, df) == pytest.approx(p, rel=1e-6)
    assert stats.student_t_two_sided_p(-t, df) == pytest.approx(p, rel=1e-6)


def test_dm_hand_computed():
    # d = [1, 2, 3], m = 1: centred [-1, 0, 1]; gamma0 = 2/3, gamma1 = 0 -> LRV = 2/3.
    # DM = 2 / sqrt((2/3) / 3) = 4.2426; HLN (h = 1): x sqrt((3 + 1 - 2) / 3) = 2 sqrt(3).
    # df = 2 has a closed form: p = 1 - t / sqrt(2 + t^2) with t = 2 sqrt(3) -> 0.0741799.
    res = stats.diebold_mariano(np.array([1.0, 2.0, 3.0]), horizon=1, bandwidth=1)
    assert res is not None
    assert res.lrv == pytest.approx(2 / 3)
    assert res.statistic == pytest.approx(2 * math.sqrt(3))
    assert res.pvalue == pytest.approx(1 - res.statistic / math.sqrt(2 + res.statistic**2))
    assert res.pvalue == pytest.approx(0.0741799, abs=1e-6)


def _brute_force_lrv(d, m):
    n = len(d)
    mean = sum(d) / n
    c = [x - mean for x in d]
    lrv = sum(x * x for x in c) / n
    for k in range(1, m + 1):
        gamma = sum(c[t] * c[t - k] for t in range(k, n)) / n
        lrv += 2 * (1 - k / (m + 1)) * gamma
    return lrv


def test_newey_west_matches_brute_force_and_caps_bandwidth():
    d = list(np.random.default_rng(3).normal(size=40).cumsum())
    assert stats.newey_west_lrv(np.array(d), 7) == pytest.approx(_brute_force_lrv(d, 7))
    assert stats.newey_west_lrv(np.array(d), 500) == pytest.approx(_brute_force_lrv(d, 39))


def test_newey_west_known_autocovariance():
    # d = [1, -1, 1, -1]: mean 0, gamma0 = 1, gamma1 = -3/4; m = 1 weight 1/2 -> 1 + 2 * 0.5 * -0.75
    assert stats.newey_west_lrv(np.array([1.0, -1, 1, -1]), 1) == pytest.approx(0.25)


def test_dm_null_for_constant_series_and_tiny_sample():
    assert stats.diebold_mariano(np.full(50, 0.3), 24, 5) is None
    assert stats.diebold_mariano(np.array([1.0, 2.0]), 1, 1) is None


def test_dm_detects_a_real_shift():
    d = np.random.default_rng(1).normal(-0.5, 1.0, size=600)
    res = stats.diebold_mariano(d, horizon=24, bandwidth=47)
    assert res is not None
    assert res.statistic < -5
    assert res.pvalue < 1e-5


def test_bootstrap_exact_small_case():
    # s = 1..4 hourly, n = 1, block 2: block sums 3, 5, 7 over 2 pairs. Two blocks per resample
    # give the estimate (a + b) / 4 in {1.5 (1/9), 2 (2/9), 2.5 (3/9), 3 (2/9), 3.5 (1/9)}.
    # The 2.5th percentile is 1.5 and the 97.5th is 3.5.
    lo, hi, est = stats.moving_block_bootstrap_ci(
        np.array([1.0, 2, 3, 4]), np.ones(4), block=2, level=0.95
    )
    assert (lo, hi) == (1.5, 3.5)
    assert est == 2.5


def test_bootstrap_level_changes_interval_and_is_seeded():
    rng = np.random.default_rng(0)
    s = rng.normal(-0.3, 1, size=500)
    n = np.ones(500)
    a = stats.moving_block_bootstrap_ci(s, n, level=0.95)
    assert a == stats.moving_block_bootstrap_ci(s, n, level=0.95)
    wide = stats.moving_block_bootstrap_ci(s, n, level=0.99)
    assert wide[0] < a[0] < a[2] < a[1] < wide[1]
    assert a[2] == pytest.approx(s.mean())


def test_bootstrap_constant_and_weighted_estimate():
    lo, hi, est = stats.moving_block_bootstrap_ci(np.full(300, 2.0), np.full(300, 2.0))
    assert lo == hi == est == 1.0
    # Pair weighting: hours with more stations count more (sum S / sum N, not a mean of S / N).
    s, n = np.array([4.0, 1.0]), np.array([4.0, 1.0])
    assert stats.moving_block_bootstrap_ci(s, n, block=1)[2] == 1.0
    s, n = np.array([8.0, 1.0]), np.array([4.0, 1.0])
    assert stats.moving_block_bootstrap_ci(s, n, block=1)[2] == pytest.approx(9 / 5)


def test_bootstrap_block_longer_than_series_uses_whole_series():
    lo, hi, est = stats.moving_block_bootstrap_ci(np.array([1.0, 3.0]), np.ones(2), block=72)
    assert lo == hi == est == 2.0


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        stats.moving_block_bootstrap_ci(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        stats.moving_block_bootstrap_ci(np.ones(3), np.ones(2))
