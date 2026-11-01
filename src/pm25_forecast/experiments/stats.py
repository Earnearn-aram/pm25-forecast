"""Inference helpers of docs/experiments/protocol.md section 6 (KAN-82). numpy and math only.

Unit of analysis: the hourly, pair-weighted series. For origin hour ``t``, ``s[t]`` is the sum of
the paired absolute-error differences (candidate - baseline) over the stations present and
``n[t]`` their count; ``s[t] / n[t]`` is the Diebold-Mariano series. Hours without pairs have
``n = 0`` (and ``s = 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 20261101
BOOTSTRAP_B = 10_000
PRIMARY_BLOCK_H = 72
SENSITIVITY_BLOCKS_H = (24, 168)
DM_PRIMARY_BANDWIDTH = 167
DM_MIN_PRESENT = 0.95


def moving_block_bootstrap_ci(
    s: np.ndarray,
    n: np.ndarray,
    *,
    block: int = PRIMARY_BLOCK_H,
    level: float = 0.95,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Percentile CI of ``sum(s) / sum(n)`` by a moving-block bootstrap on the hourly grid.

    Blocks are ``block`` consecutive grid hours starting uniformly at any hour where one fits
    (a series shorter than ``block`` uses one block of its full length). ``ceil(G / block)``
    blocks are drawn with replacement per resample. Returns ``(low, high, estimate)``.
    """
    s = np.asarray(s, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if s.shape != n.shape or s.ndim != 1 or s.size == 0:
        raise ValueError("s and n must be equal-length, non-empty 1-D arrays")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if n.sum() <= 0:
        raise ValueError("no pairs")
    grid = s.size
    block = min(block, grid)
    n_blocks = math.ceil(grid / block)
    cs = np.concatenate(([0.0], np.cumsum(s)))
    cn = np.concatenate(([0.0], np.cumsum(n)))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, grid - block + 1, size=(b, n_blocks))
    num = (cs[starts + block] - cs[starts]).sum(axis=1)
    den = (cn[starts + block] - cn[starts]).sum(axis=1)
    ok = den > 0
    est = num[ok] / den[ok]
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(est, [tail, 100 - tail])
    return float(low), float(high), float(s.sum() / n.sum())


def newey_west_lrv(d: np.ndarray, bandwidth: int) -> float:
    """Long-run variance of ``d`` with Bartlett weights ``1 - k / (m + 1)``, lags ``1..m``.

    ``gamma_k`` divides by ``len(d)`` (not ``len(d) - k``), which keeps the estimate non-negative
    in theory; ``m`` is capped at ``len(d) - 1``.
    """
    d = np.asarray(d, dtype=np.float64)
    size = d.size
    centered = d - d.mean()
    m = min(bandwidth, size - 1)
    lrv = float(centered @ centered) / size
    for k in range(1, m + 1):
        lrv += 2 * (1 - k / (m + 1)) * float(centered[k:] @ centered[:-k]) / size
    return lrv


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """``P(|T| >= |t|)`` for Student's t with ``df`` degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        raise ValueError("need df > 0 and a finite statistic")
    return _betainc(df / 2, 0.5, df / (df + t * t))


@dataclass(frozen=True)
class DMResult:
    statistic: float  # after the Harvey-Leybourne-Newbold correction
    pvalue: float
    lrv: float
    n: int
    bandwidth: int


def diebold_mariano(d: np.ndarray, horizon: int, bandwidth: int) -> DMResult | None:
    """DM test of ``E[d] = 0`` (two-sided) with Newey-West variance and the HLN correction.

    ``d`` is the loss differential series (here the hourly ``s / n``), ``horizon`` the forecast
    step ``h`` of the HLN factor ``sqrt((n + 1 - 2h + h(h-1)/n) / n)``; the p-value uses Student-t
    with ``n - 1`` df. Returns None when the variance is not positive or the factor is undefined.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 3:
        return None
    lrv = newey_west_lrv(d, bandwidth)
    # A constant series gives rounding noise rather than exactly 0: treat that as no variance.
    if not lrv > 1e-18 * max(1.0, float(np.abs(d).max())) ** 2:
        return None
    radicand = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if radicand <= 0:
        return None
    stat = float(d.mean()) / math.sqrt(lrv / n) * math.sqrt(radicand)
    return DMResult(stat, student_t_two_sided_p(stat, n - 1), lrv, n, bandwidth)
