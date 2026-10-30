"""Shared helpers for the KAN-65 offline experiments (not part of the serving path)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np

STATIONS = ("02t", "59t", "61t", "bkp100t", "bkp101t")
EXPECTED_SHA = "fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09"
HORIZON = 24
LAG = 1  # reporting lag in hours (servable)
HIGH = 37.5
BOOT_SEED = 20261030
MODEL_SEED = 20261029
GAP_START = datetime(2026, 5, 19, tzinfo=UTC)
GAP_END = datetime(2026, 5, 22, tzinfo=UTC)  # exclusive
GAP_STATIONS = ("bkp100t", "bkp101t")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Grid:
    """Dense hourly UTC grid, one NaN-filled array per station (no imputation)."""

    def __init__(self, path: Path):
        self.sha = sha256(path)
        with duckdb.connect(str(path), read_only=True) as con:
            con.execute("SET TimeZone = 'UTC'")
            rows = con.execute(
                "SELECT station_id, ts, pm25, source FROM observations "
                "WHERE pm25 IS NOT NULL ORDER BY station_id, ts"
            ).fetchall()
        all_ts = [r[1].astimezone(UTC) for r in rows]
        self.t0 = min(all_ts).replace(minute=0, second=0, microsecond=0)
        self.t_last = max(all_ts)
        self.n = int((self.t_last - self.t0) / timedelta(hours=1)) + 1
        self.values = {s: np.full(self.n, np.nan) for s in STATIONS}
        self.source = {s: np.zeros(self.n, dtype=np.int8) for s in STATIONS}  # 1 openaq 2 air4thai
        for (st, ts, v, src), t in zip(rows, all_ts, strict=True):
            if not np.isfinite(v) or v < 0 or st not in self.values:
                continue
            i = int((t - self.t0) / timedelta(hours=1))
            self.values[st][i] = v
            self.source[st][i] = 1 if src == "openaq" else 2

    def idx(self, t: datetime) -> int:
        return int((t - self.t0) / timedelta(hours=1))

    def time(self, i: int) -> datetime:
        return self.t0 + timedelta(hours=int(i))


def day_block_ci(
    diff: np.ndarray, days: np.ndarray, n_boot: int = 10000, seed: int = BOOT_SEED,
    block: int = 1, level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean paired difference with percentile CI from a block bootstrap over target days.

    ``block`` = number of consecutive days per block (1 = day-block; 7 = week-block). Weighted by
    pooled station-hours: resampled mean = sum(diff)/count over drawn blocks.
    """
    uniq = np.unique(days)
    nb = int(np.ceil(len(uniq) / block))
    day_pos = np.searchsorted(uniq, days)
    blk = day_pos // block
    sums = np.bincount(blk, weights=diff, minlength=nb)
    cnts = np.bincount(blk, minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nb, size=(n_boot, nb))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    a = (1 - level) / 2
    return float(diff.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
