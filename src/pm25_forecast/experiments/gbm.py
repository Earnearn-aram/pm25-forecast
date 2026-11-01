"""KAN-89 (E12): pooled LightGBM model factory for ``experiments.evaluate``.

    python -m pm25_forecast.experiments.evaluate --prereg docs/experiments/prereg/<id>.yaml \
        --snapshot <dir> --model pm25_forecast.experiments.gbm:fit

One LightGBM per horizon, pooled over stations (the station is a categorical feature). All feature
logic is ``pm25_forecast.features.library_v2`` (``prepare_series`` once per fold, then
``build_features_v2``); this file only selects columns, trains and tunes.

Params (``prereg.params``): ``feature_set`` (registry name), ``objective`` (l2 | l1 | huber),
``target`` (level | delta; delta = value minus the newest reading in [t-3h, t-1h], the persistence
anchor), ``drop_groups`` (subset of lags, rolling, calendar, cross), ``seed``, ``budget`` (number of
hyper-parameter configurations tried per fit, the first one is the default), ``inner_folds``,
``huber_alpha``, ``num_threads``, ``log_dir`` / ``log_name`` (tuning log written as JSON).

Tuning happens inside ``fit`` on the training part only: a seeded random search over a fixed
grid, scored by MAE on the level scale in an expanding-window inner CV over the training pairs,
purged by the horizon (a training label must be known at the validation origin).

Cross-station features. The runner's predictor sees one station per call, so peers are read from a
memo of the snapshots seen so far (seeded from the training data). To be independent of the
station order within an origin, the cross-station group is evaluated one hour earlier
(origin - 1 h), in training and in prediction alike, and reported as ``<name>__l1``. The
nearest-station feature is not used (the snapshot has no station coordinates).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features.library_v2 import (
    ALL_FEATURE_NAMES,
    CALENDAR_NAMES,
    CROSS_STATION_NAMES,
    DIFF_NAMES,
    EWMA_NAME,
    LAG_NAMES,
    MISSING_NAMES,
    ROLLING_NAMES,
    PreparedSeries,
    build_features_v2,
    feature_set_names,
    prepare_series,
)

HOUR = timedelta(hours=1)
USED_CROSS = tuple(n for n in CROSS_STATION_NAMES if n != "nearest_station_pm25")
BASE_ALL = tuple(n for n in ALL_FEATURE_NAMES if n not in CROSS_STATION_NAMES)
OWN_TAIL = 80  # own readings kept at prediction time (covers the 72 h windows)
PEER_TAIL = 12  # peer readings kept in the memo (cross features need origin-4h .. origin-2h)
ANCHOR_LAGS = ("pm25_lag_1h", "pm25_lag_2h", "pm25_lag_3h")

GROUPS: dict[str, frozenset[str]] = {
    "lags": frozenset(LAG_NAMES + DIFF_NAMES + tuple(n for n in MISSING_NAMES if "lag" in n)),
    "rolling": frozenset(
        (*ROLLING_NAMES, EWMA_NAME, *(n for n in MISSING_NAMES if n.startswith("n_valid")))
    ),
    "calendar": frozenset(CALENDAR_NAMES),
    "cross": frozenset(USED_CROSS),
}
for _w in (3, 6, 12, 24, 72):  # fs_v2_core uses plain window means: part of the rolling group
    GROUPS["rolling"] |= {f"pm25_mean_{_w}h"}

LGB_GRID: dict[str, tuple[Any, ...]] = {
    "num_leaves": (4, 7, 15, 31),
    "learning_rate": (0.03, 0.06, 0.1),
    "n_estimators": (100, 200, 300),
    "min_child_samples": (20, 50, 100),
    "feature_fraction": (0.6, 0.8, 1.0),
    "lambda_l2": (0.0, 1.0, 10.0),
}
DEFAULT_CONFIG: dict[str, Any] = {
    "num_leaves": 7,
    "learning_rate": 0.06,
    "n_estimators": 200,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
}


def selected_names(feature_set: str, drop_groups: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Features of the registered set minus the dropped groups and minus nearest_station_pm25."""
    drop: set[str] = set()
    for g in drop_groups:
        if g not in GROUPS:
            raise KeyError(f"unknown group {g!r}; known: {sorted(GROUPS)}")
        drop |= GROUPS[g]
    return tuple(
        n for n in feature_set_names(feature_set) if n not in drop and n != "nearest_station_pm25"
    )


def config_grid(budget: int, seed: int) -> list[dict[str, Any]]:
    """The default configuration, then ``budget - 1`` seeded random draws from ``LGB_GRID``."""
    rng = np.random.default_rng(seed)
    configs = [dict(DEFAULT_CONFIG)]
    while len(configs) < budget:
        configs.append({k: v[int(rng.integers(len(v)))] for k, v in LGB_GRID.items()})
    return configs


def _all_features(
    prep: PreparedSeries, station: str, origin: datetime, valid: datetime
) -> dict[str, float]:
    """Every base feature at ``origin`` plus the cross group at ``origin - 1 h`` (NaN = missing)."""
    base = build_features_v2(prep, station, origin, valid, names=BASE_ALL)
    cross = build_features_v2(prep, station, origin - HOUR, valid, names=USED_CROSS)
    out = {k: (math.nan if v is None else float(v)) for k, v in base.items()}
    out.update({f"{k}__l1": (math.nan if v is None else float(v)) for k, v in cross.items()})
    return out


def _columns(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{n}__l1" if n in USED_CROSS else n for n in names)


def _anchor(feats: Mapping[str, float]) -> float | None:
    for k in ANCHOR_LAGS:
        v = feats[k]
        if not math.isnan(v):
            return v
    return None


# Training features are a pure function of (station, origin, valid) and of rows that were already
# available at the origin, so within one training snapshot every (station, origin, valid) is
# computed once. The cache belongs to ONE snapshot: it is keyed by a fingerprint of the training
# rows and of the feature list, and replaced as soon as another snapshot (or another feature
# library) is fitted, so it can never serve features built from different data.
_TRAIN_CACHE: dict[tuple[str, datetime, datetime], dict[str, float]] = {}
_TRAIN_CACHE_FP: list[str | None] = [None]


def _fingerprint(observations: Any) -> str:
    h = hashlib.sha256()
    h.update(repr((BASE_ALL, USED_CROSS)).encode())
    for o in sorted(observations, key=lambda o: (o.station_id, o.ts)):
        h.update(
            f"{o.station_id}|{o.ts.isoformat()}|{o.value!r}|{o.available_at.isoformat()};".encode()
        )
    return h.hexdigest()


def _train_cache_for(observations: Any) -> dict[tuple[str, datetime, datetime], dict[str, float]]:
    fp = _fingerprint(observations)
    if _TRAIN_CACHE_FP[0] != fp:
        _TRAIN_CACHE.clear()
        _TRAIN_CACHE_FP[0] = fp
    return _TRAIN_CACHE


def inner_cv_masks(tgt: np.ndarray, h: int, inner_folds: int):
    """Expanding-window inner CV over target timestamps (epoch seconds), purged by the horizon.

    A validation pair with target ``v`` is forecast at origin ``v - h``; a training label with
    target ``t`` becomes available at ``t + 1 h`` (protocol: ``available_at = ts + 1 h``), so it is
    known at the validation origin only if ``t + 1 h <= v - h``. The fold boundary ``edges[j]`` is
    the earliest validation target, hence ``t <= edges[j] - (h + 1) h``.
    """
    edges = np.quantile(tgt, np.linspace(0, 1, inner_folds + 2))
    for j in range(1, inner_folds + 1):
        val = (tgt >= edges[j]) & (tgt < edges[j + 1] if j < inner_folds else tgt <= edges[-1])
        tr = tgt <= edges[j] - (h + 1) * 3600
        yield tr, val


def _lgb_params(cfg: Mapping[str, Any], objective: str, alpha: float, seed: int, threads: int):
    p: dict[str, Any] = {
        "objective": {"l2": "regression", "l1": "regression_l1", "huber": "huber"}[objective],
        "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"],
        "min_data_in_leaf": cfg["min_child_samples"],
        "feature_fraction": cfg["feature_fraction"],
        "lambda_l2": cfg["lambda_l2"],
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
    }
    if objective == "huber":
        p["alpha"] = alpha
    return p


def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
    import lightgbm as lgb

    prm = dict(params or {})
    feature_set = str(prm.get("feature_set", "fs_v2_core"))
    objective = str(prm.get("objective", "l2"))
    target_kind = str(prm.get("target", "level"))
    if objective not in ("l2", "l1", "huber") or target_kind not in ("level", "delta"):
        raise ValueError(f"bad objective/target: {objective}/{target_kind}")
    seed = int(prm.get("seed", 20261101))
    budget = int(prm.get("budget", 8))
    inner_folds = int(prm.get("inner_folds", 3))
    alpha = float(prm.get("huber_alpha", 8.0))
    threads = int(prm.get("num_threads", 4))
    names = selected_names(feature_set, tuple(prm.get("drop_groups", ())))
    cols = _columns(names)
    stations = tuple(sorted(train.stations))
    code = {s: i for i, s in enumerate(stations)}
    configs = config_grid(budget, seed)
    delta = target_kind == "delta"

    train_cache = _train_cache_for(train.observations)
    by_station: dict[str, dict[datetime, Observation]] = defaultdict(dict)
    for o in train.observations:
        by_station[o.station_id][o.ts] = o
    prep = prepare_series({s: list(rows.values()) for s, rows in by_station.items()})

    def to_row(station: str, feats: Mapping[str, float]) -> list[float]:
        return [float(code[station])] + [feats[c] for c in cols]

    models: dict[int, Any] = {}
    log: list[dict[str, Any]] = []
    for h in train.horizons:
        rows: list[list[float]] = []
        ys: list[float] = []
        anchors: list[float] = []
        targets: list[datetime] = []
        for s in stations:
            for target, label in by_station.get(s, {}).items():
                origin = target - h * HOUR
                key = (s, origin, target)
                feats = train_cache.get(key)
                if feats is None:
                    feats = _all_features(prep, s, origin, target)
                    train_cache[key] = feats
                anchor = _anchor(feats)
                if anchor is None:
                    continue  # never scored: persistence has no value either
                rows.append(to_row(s, feats))
                ys.append(label.value)
                anchors.append(anchor)
                targets.append(target)
        if len(rows) < 200:
            continue
        order = np.argsort([t.timestamp() for t in targets], kind="stable")
        x = np.asarray(rows, dtype=float)[order]
        y = np.asarray(ys)[order]
        a = np.asarray(anchors)[order]
        tgt = np.asarray([targets[i].timestamp() for i in order])
        z = y - a if delta else y

        def train_booster(cfg, idx, x=x, z=z):
            ds = lgb.Dataset(x[idx], z[idx], categorical_feature=[0], free_raw_data=False)
            return lgb.train(
                _lgb_params(cfg, objective, alpha, seed, threads),
                ds,
                num_boost_round=int(cfg["n_estimators"]),
            )

        # Inner time-series CV on the training part only: expanding window, purged by h.
        scores = np.zeros(len(configs))
        n_used = 0
        for tr, val in inner_cv_masks(tgt, h, inner_folds):
            if tr.sum() < 100 or val.sum() < 20:
                continue
            n_used += 1
            for i, cfg in enumerate(configs):
                b = train_booster(cfg, np.flatnonzero(tr))
                pred = b.predict(x[val]) + (a[val] if delta else 0.0)
                scores[i] += float(np.mean(np.abs(pred - y[val])))
        if n_used:
            scores /= n_used
            best = int(np.argmin(scores))
        else:
            best = 0
            scores = np.full(len(configs), math.nan)
        final = train_booster(configs[best], np.arange(len(y)))
        models[h] = final
        gain = final.feature_importance("gain")
        log.append(
            {
                "refit_time": train.refit_time.isoformat(),
                "horizon": h,
                "n_train": len(y),
                "inner_folds_used": n_used,
                "chosen": best,
                "config": configs[best],
                "inner_mae": [None if math.isnan(v) else round(float(v), 4) for v in scores],
                "gain": {
                    n: round(float(g), 3) for n, g in zip(("station", *cols), gain, strict=True)
                },
            }
        )
    _write_log(prm, log)

    # ---- prediction -------------------------------------------------------------------------
    memo: dict[str, dict[datetime, Observation]] = {s: {} for s in stations}
    for s, rows_by_ts in by_station.items():
        for ts in sorted(rows_by_ts)[-PEER_TAIL:]:
            memo[s][ts] = rows_by_ts[ts]
    cache: dict[str, Any] = {"key": None, "prep": None}

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        booster = models.get(horizon)
        s = snapshot.station_id
        if booster is None or s not in code:
            return None
        origin = snapshot.prediction_time
        own = snapshot.observations[-OWN_TAIL:]
        peers = memo.setdefault(s, {})
        for o in own[-PEER_TAIL:]:
            peers[o.ts] = o
        for ts in sorted(peers)[:-PEER_TAIL]:
            del peers[ts]
        key = (s, origin, len(own), own[-1].ts if own else None)
        if cache["key"] != key:
            records = {p: list(m.values()) for p, m in memo.items() if p != s}
            records[s] = list(own)
            cache["key"], cache["prep"] = key, prepare_series(records)
        feats = _all_features(cache["prep"], s, origin, origin + horizon * HOUR)
        anchor = _anchor(feats)
        if anchor is None:
            return None
        pred = float(booster.predict(np.asarray([to_row(s, feats)], dtype=float))[0])
        return pred + anchor if delta else pred

    return predict


def _write_log(prm: Mapping[str, Any], log: list[dict[str, Any]]) -> None:
    log_dir, name = prm.get("log_dir"), prm.get("log_name")
    if not log_dir or not name:
        return
    path = Path(str(log_dir)).expanduser() / f"{name}.tuning.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent: a re-run replaces the entries of the same refit time instead of appending twice.
    new = {e["refit_time"] for e in log}
    keep = []
    if path.exists():
        keep = [
            ln
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln)["refit_time"] not in new
        ]
    keep += [json.dumps(entry, sort_keys=True) for entry in log]
    path.write_text("\n".join(keep) + "\n", encoding="utf-8")
