"""Leakage-safe feature table for the KAN-65 experiments.

Reference time r = origin - 1 h (reporting lag). Every feature reads only grid indices <= o - 1,
except calendar features, which use the target timestamp (known at issue time). Missing inputs stay
NaN; no sample is imputed. Rows need a target and a persistence value (latest reading within
[origin-3h, origin-1h]), the KAN-15 / KAN-64 pair rule.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import GAP_END, GAP_START, GAP_STATIONS, HORIZON, STATIONS, Grid

BKK = timedelta(hours=7)  # local-time shift used only for calendar features / EDA labels

SET_FEATURES: dict[str, list[str]] = {}
_A = ["pm25_latest"]
_B = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "station_id",
]
_C = ["pm25_t24", "pm25_t48", "pm25_t72", "pm25_t168"]
_D = [
    "mean3",
    "max3",
    "std3",
    "mean6",
    "max6",
    "std6",
    "mean24",
    "max24",
    "std24",
    "trend3",
    "trend6",
]
_E = ["nbr_latest_mean", "nbr_mean24"]
EXTRA = ["src_air4thai", "leak_nbr_target"]
SET_FEATURES["A"] = _A
SET_FEATURES["B"] = _A + _B
SET_FEATURES["C"] = _A + _B + _C
SET_FEATURES["D"] = _A + _B + _C + _D
SET_FEATURES["E"] = _A + _B + _C + _D + _E
ALL_FEATURES = SET_FEATURES["E"]


@dataclass
class Table:
    station: np.ndarray  # str
    origin_idx: np.ndarray
    target_idx: np.ndarray
    y: np.ndarray
    latest: np.ndarray
    feats: dict[str, np.ndarray]
    target_day: np.ndarray  # int days since epoch of the target UTC date
    is_train: np.ndarray
    is_test: np.ndarray
    source_target: np.ndarray
    test_start_idx: int
    grid: Grid

    def X(self, names: list[str]) -> np.ndarray:
        return np.column_stack([self.feats[n] for n in names])

    def mask(self, m: np.ndarray) -> Table:
        return Table(
            self.station[m],
            self.origin_idx[m],
            self.target_idx[m],
            self.y[m],
            self.latest[m],
            {k: v[m] for k, v in self.feats.items()},
            self.target_day[m],
            self.is_train[m],
            self.is_test[m],
            self.source_target[m],
            self.test_start_idx,
            self.grid,
        )


def _at(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(idx.shape, np.nan)
    ok = (idx >= 0) & (idx < len(a))
    out[ok] = a[idx[ok]]
    return out


def _latest(a: np.ndarray, o: np.ndarray) -> np.ndarray:
    res = np.full(o.shape, np.nan)
    for k in (3, 2, 1):  # prefer the newest (k=1)
        v = _at(a, o - k)
        res = np.where(np.isnan(v), res, v)
    # loop goes 3,2,1 and overwrites, so newest non-NaN wins
    return res


def _window(a: np.ndarray, o: np.ndarray, w: int):
    """Stats over grid indices o-w .. o-1 (ends at r = o-1)."""
    pad = np.concatenate([np.full(w, np.nan), a])
    win = sliding_window_view(pad, w)  # win[j] covers pad[j:j+w] == a[j-w:j]
    sel = win[o]  # a[o-w:o]
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(sel), axis=1)
        mean = np.where(cnt > 0, np.nansum(sel, axis=1) / np.maximum(cnt, 1), np.nan)
        mx = np.where(cnt > 0, np.nanmax(np.where(np.isnan(sel), -np.inf, sel), axis=1), np.nan)
        dev = np.where(np.isnan(sel), 0.0, sel - mean[:, None])
        std = np.where(cnt >= 2, np.sqrt((dev**2).sum(1) / np.maximum(cnt - 1, 1)), np.nan)
    mx = np.where(np.isfinite(mx), mx, np.nan)
    return mean, mx, std


def build_table(grid: Grid, horizon: int = HORIZON) -> Table:
    first = grid.t0
    last_origin = grid.t_last - timedelta(hours=HORIZON)
    origin_count = int((last_origin - first) / timedelta(hours=1)) + 1
    test_start_idx = int(origin_count * 0.85)  # KAN-15 rule; index == hours after t0
    origins = np.arange(origin_count)

    lat = {s: _latest(grid.values[s], origins) for s in STATIONS}
    m24 = {s: _window(grid.values[s], origins, 24)[0] for s in STATIONS}

    src_f = {
        s: np.where(grid.source[s] == 0, np.nan, grid.source[s].astype(float)) for s in STATIONS
    }
    srclat = {s: _latest(src_f[s], origins) for s in STATIONS}
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ALL_FEATURES + EXTRA}
    st_l, o_l, t_l, y_l, l_l, src_l = [], [], [], [], [], []
    for si, s in enumerate(STATIONS):
        a = grid.values[s]
        tgt = origins + horizon
        y = _at(a, tgt)
        keep = ~np.isnan(y) & ~np.isnan(lat[s])
        if s in GAP_STATIONS:  # assert the known gap never reaches a sample
            lo, hi = grid.idx(GAP_START), grid.idx(GAP_END)
            assert not np.any(keep & (tgt >= lo) & (tgt < hi))
        o = origins[keep]
        t = tgt[keep]
        f: dict[str, np.ndarray] = {}
        f["pm25_latest"] = lat[s][keep]
        tl = [grid.time(i) + BKK for i in t]
        hour = np.array([x.hour for x in tl], float)
        dow = np.array([x.weekday() for x in tl], float)
        mon = np.array([x.month - 1 for x in tl], float)
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
        f["station_id"] = np.full(len(o), float(si))
        for name, back in (
            ("pm25_t24", 24),
            ("pm25_t48", 48),
            ("pm25_t72", 72),
            ("pm25_t168", 168),
        ):
            idx = t - back
            f[name] = _at(a, np.where(idx <= o - 1, idx, -1))  # as-of: reading must be <= r
        for w in (3, 6, 24):
            mean, mx, std = _window(a, o, w)
            f[f"mean{w}"], f[f"max{w}"], f[f"std{w}"] = mean, mx, std
        f["trend3"] = f["pm25_latest"] - _at(a, o - 1 - 3)
        f["trend6"] = f["pm25_latest"] - _at(a, o - 1 - 6)
        others = [x for x in STATIONS if x != s]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nl = np.column_stack([lat[x][keep] for x in others])
            nm = np.column_stack([m24[x][keep] for x in others])
            f["nbr_latest_mean"] = np.where(
                np.isnan(nl).all(1), np.nan, np.nanmean(np.where(np.isnan(nl), np.nan, nl), 1)
            )
            f["nbr_mean24"] = np.where(np.isnan(nm).all(1), np.nan, np.nanmean(nm, 1))
        # flag: the latest reading comes from the air4thai feed (live feed = 1)
        f["src_air4thai"] = (srclat[s][keep] == 2).astype(float)
        # DELIBERATELY LEAKY control (never a candidate): neighbours' reading at the target hour
        nt = np.column_stack([_at(grid.values[x], t) for x in others])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f["leak_nbr_target"] = np.nanmean(nt, 1)
        for k in ALL_FEATURES + EXTRA:
            cols[k].append(f[k])
        st_l.append(np.full(len(o), s))
        o_l.append(o)
        t_l.append(t)
        y_l.append(y[keep])
        l_l.append(lat[s][keep])
        src_l.append(grid.source[s][t])
    cat = np.concatenate
    o_all, t_all = cat(o_l), cat(t_l)
    order = np.lexsort((cat(st_l), o_all))  # chronological by origin, then station
    take = lambda arr: arr[order]  # noqa: E731
    o_all, t_all = take(o_all), take(t_all)
    base_day = int(grid.t0.replace(hour=0).timestamp() // 86400)
    day = base_day + (t_all + grid.t0.hour) // 24
    return Table(
        station=take(cat(st_l)),
        origin_idx=o_all,
        target_idx=t_all,
        y=take(cat(y_l)),
        latest=take(cat(l_l)),
        feats={k: take(cat(v)) for k, v in cols.items()},
        target_day=day,
        is_train=t_all < test_start_idx,
        is_test=o_all >= test_start_idx,
        source_target=take(cat(src_l)),
        test_start_idx=test_start_idx,
        grid=grid,
    )
