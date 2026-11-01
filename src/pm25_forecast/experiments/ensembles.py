"""KAN-92 (E15) ensembles of the family-best models, as one factory for ``experiments.evaluate``.

    python -m pm25_forecast.experiments.evaluate --prereg docs/experiments/prereg/<id>.yaml \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 \
        --model pm25_forecast.experiments.ensembles:fit

The factory is called once per weekly fold. It fits its member models on the same train snapshot
and combines their predictions at predict time. The members are the family-best configurations of
E05, E06, E07, E11 and E12 exactly as preregistered there (``COMPONENT_PARAMS``; nothing is
re-tuned). ``params["variant"]`` selects the combination:

- ``equal``: plain average of the five members.
- ``invmae``: weights proportional to ``1 / MAE`` of each member on the most recent
  ``weights_window_days`` days of out-of-fold predictions (matured targets only, see below).
- ``stack``: non-negative least squares with intercept (a linear stacker) on all out-of-fold
  predictions of the five members, pooled over stations.
- ``smoother_gbm``: 0.5 * smoother + 0.5 * the E12 LightGBM without the cross-station group
  (no fitted weights).
- ``ridge_clim``: 0.5 * ridge + 0.5 * climatology (the two best simple members, no fitted weights).

Out-of-fold predictions (``invmae`` and ``stack`` only). Everything happens inside one outer
fold's training part: the inner splits come from ``gbm.inner_cv_masks`` over the target times of
the training pairs (expanding window, ``inner_folds`` validation blocks, a training label is used
only if its target is at least ``horizon + 1`` h before the first validation target, because a
label with target t is available at t + 1 h). For each inner split every member is refitted on
the matured rows of that split alone (a ``TrainSnapshot`` with the inner refit time) and predicts
the inner validation pairs from snapshots built with the serving lag. Nothing from the outer
validation fold, and no label that was not yet available at the inner validation origin, enters.
The weights are fixed at the outer refit time, like every other parameter (no hourly updates).

Serving lag and statelessness. The predictor does not trust the snapshot it gets: it rebuilds it
with only rows that have ``ts <= origin - 1 h`` and ``available_at <= origin`` (by time, never by
position) before it calls a member. This is what the serving-lag probe needs, because several
members slice ``snapshot.observations`` by position (see ``linear_models.py`` for the same fix).
The combination step reads no state. A member that returns None makes the ensemble return None
(rows are never imputed and the ensemble scores the same rows for every variant).

Side effects of the members are redirected so that they cannot overwrite the artefacts of the
earlier tickets: the KAN-84 coefficient dump and the KAN-88 prediction CSV go to
``$PM25_KAN92_LOG_DIR`` (default ``~/pm25-experiments/cache/kan92``); the KAN-89 tuning log is not
requested. Diagnostics of every fit (weights, out-of-fold MAE per member) are written there as JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot, persistence_value

HOUR = timedelta(hours=1)
SEED = 20261101
MEMBER_ORDER = ("ridge", "clim", "gbm", "arima", "wx")
VARIANT_MEMBERS: dict[str, tuple[str, ...]] = {
    "equal": MEMBER_ORDER,
    "invmae": MEMBER_ORDER,
    "stack": MEMBER_ORDER,
    "smoother_gbm": ("smoother", "gbm"),
    "ridge_clim": ("ridge", "clim"),
}
FITTED_VARIANTS = ("invmae", "stack")  # need out-of-fold predictions
DEFAULTS: dict[str, Any] = {
    "variant": "equal",
    "inner_folds": 3,
    "weights_window_days": 14,
    "min_oof_rows": 200,
}

# Frozen parameters of the family-best members: copies of the ``params`` of these preregistrations
# (a test compares them with the YAML files). Names that only route diagnostics are changed.
COMPONENT_PARAMS: dict[str, dict[str, Any]] = {
    # e06-ridge-pool-raw-chg (KAN-84): pooled ridge, raw target, change from latest, fs_v2_core
    "ridge": {
        "variant": "e15-ridge-pool-raw-chg",
        "model": "ridge",
        "scope": "pooled",
        "target": "raw",
        "output": "change",
        "feature_set": "fs_v2_core",
    },
    # e05-clim-28d-blend (KAN-83): same-hour climatology of 28 days, equal blend with persistence
    "clim": {"days": 28, "persistence_weight": 0.5},
    # e12-abl-no-cross (KAN-89): the E12 best (e12-full-l2-delta: pooled LightGBM, fs_v2_full, L2,
    # delta target, budget 8) without the cross-station group. The full model reads the other
    # stations from a memo of earlier calls (state), so a fresh instance predicts differently and
    # the serving-lag probe is inconclusive; the no-cross twin is stateless (MAE 3.874 vs 3.868).
    "gbm": {
        "feature_set": "fs_v2_full",
        "objective": "l2",
        "target": "delta",
        "drop_groups": ["cross"],
        "budget": 8,
        "inner_folds": 3,
        "huber_alpha": 8.0,
        "num_threads": 4,
        "seed": SEED,
    },
    # e07-arima-d1 (KAN-85): per-station ARIMA(p,1,q), order by AIC
    "arima": {
        "name": "arima-d1",
        "factory": "arima",
        "d": 1,
        "p_grid": [0, 1, 2, 3],
        "q_grid": [0, 1, 2],
        "maxiter": 50,
        "stateless": True,  # runtime flag, not a model change (differs from E07 by about 1e-7)
    },
    # e11-lgbm-wx-d2 (KAN-88): LightGBM with weather forecasts issued 2 days ahead
    # (previous_day2: strictly available at the origin, protocol section 2)
    "wx": {
        "objective": "l1",
        "num_leaves": 7,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "min_child_samples": 50,
        "feature_fraction": 0.9,
        "seed": SEED,
        "n_seeds": 5,
    },
}
COMPONENT_PREREG = {
    "ridge": "e06-ridge-pool-raw-chg",
    "clim": "e05-clim-28d-blend",
    "gbm": "e12-abl-no-cross",
    "arima": "e07-arima-d1",
    "wx": "e11-lgbm-wx-d2",
}

_OOF_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def log_dir() -> Path:
    return Path(os.environ.get("PM25_KAN92_LOG_DIR", "~/pm25-experiments/cache/kan92")).expanduser()


def clear_cache() -> None:
    _OOF_CACHE.clear()


# ---- members -------------------------------------------------------------------------------


def member_factory(name: str) -> Callable[[TrainSnapshot], Predictor]:
    """A one-argument factory for a member (imports are lazy: the experiments group is optional)."""
    d = log_dir()
    if name == "smoother":
        from pm25_forecast.experiments.evaluate import smoother

        return smoother
    params = dict(COMPONENT_PARAMS[name])
    if name == "ridge":
        os.environ["PM25_KAN84_LOG_DIR"] = str(d / "ridge")
        from pm25_forecast.experiments.linear_models import fit as fit_ridge

        return lambda train: fit_ridge(train, params)
    if name == "clim":
        from pm25_forecast.experiments.smoothing import climatology

        return lambda train: climatology(train, params)
    if name == "gbm":
        from pm25_forecast.experiments.gbm import fit as fit_gbm

        return lambda train: fit_gbm(train, params)
    if name == "arima":
        from pm25_forecast.experiments.timeseries import arima

        return lambda train: arima(train, params)
    if name == "wx":
        os.environ["PM25_KAN88_PRED_DIR"] = str(d / "wx-preds")
        from pm25_forecast.experiments.kan88_models import lgbm_wx_d2

        return lambda train: lgbm_wx_d2(train, params)
    raise KeyError(f"unknown member {name!r}")


def clean_snapshot(snapshot: Snapshot) -> Snapshot:
    """Only rows the serving lag allows (by time): ts <= origin - 1 h and available_at <= origin."""
    origin = snapshot.prediction_time
    last = origin - HOUR
    return Snapshot(
        origin,
        snapshot.station_id,
        tuple(o for o in snapshot.observations if o.ts <= last and o.available_at <= origin),
        tuple(
            v
            for v in snapshot.vintages
            if v.issue_time <= origin and v.available_at <= origin and v.valid_time > last
        ),
    )


# ---- combination arithmetic (pure) ---------------------------------------------------------


def inverse_mae_weights(
    y: np.ndarray, preds: np.ndarray, target_s: np.ndarray, window_days: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Weights proportional to 1/MAE over the newest ``window_days`` of the given (matured) rows.

    Returns (weights, MAE per member, rows used). Only rows with a finite prediction from every
    member count; ``target_s`` is the target time in epoch seconds.
    """
    ok = np.isfinite(preds).all(axis=1)
    y, preds, target_s = y[ok], preds[ok], target_s[ok]
    if len(y) == 0:
        raise ValueError("no out-of-fold rows")
    recent = target_s >= target_s.max() - window_days * 86400.0
    err = np.abs(preds[recent] - y[recent, None]).mean(axis=0)
    inv = 1.0 / np.maximum(err, 1e-6)
    return inv / inv.sum(), err, int(recent.sum())


def fit_stacker(y: np.ndarray, preds: np.ndarray) -> tuple[float, np.ndarray]:
    """Non-negative least squares with an intercept. Returns (intercept, coefficients)."""
    from sklearn.linear_model import LinearRegression

    ok = np.isfinite(preds).all(axis=1)
    model = LinearRegression(positive=True, fit_intercept=True).fit(preds[ok], y[ok])
    return float(model.intercept_), np.asarray(model.coef_, dtype=float)


def combine(preds: Sequence[float], weights: Sequence[float], intercept: float = 0.0) -> float:
    return float(intercept + np.dot(np.asarray(weights, dtype=float), np.asarray(preds)))


# ---- out-of-fold predictions ---------------------------------------------------------------


def fingerprint(train: TrainSnapshot) -> str:
    h = hashlib.sha256()
    h.update(repr((train.refit_time.isoformat(), tuple(train.stations))).encode())
    for o in sorted(train.observations, key=lambda o: (o.station_id, o.ts)):
        h.update(
            f"{o.station_id}|{o.ts.isoformat()}|{o.value!r}|{o.available_at.isoformat()};".encode()
        )
    return h.hexdigest()


def _sub_train(train: TrainSnapshot, refit: datetime, horizon: int) -> TrainSnapshot:
    return TrainSnapshot(
        refit,
        train.stations,
        (horizon,),
        tuple(o for o in train.observations if o.ts <= refit and o.available_at <= refit),
        (),
    )


def oof_predictions(
    train: TrainSnapshot, names: tuple[str, ...], horizon: int, inner_folds: int
) -> dict[str, np.ndarray]:
    """Out-of-fold member predictions on the training part of one outer fold.

    Returns arrays ``y`` (n,), ``target_s`` (n,), ``preds`` (n, len(names)); NaN where a member
    returns None. Cached in the process by (train fingerprint, members, horizon, inner folds).
    """
    from pm25_forecast.experiments.gbm import inner_cv_masks

    key = (fingerprint(train), names, horizon, inner_folds)
    if key in _OOF_CACHE:
        return _OOF_CACHE[key]
    sorted_obs: dict[str, list[Observation]] = defaultdict(list)
    for o in sorted(train.observations, key=lambda r: r.ts):
        sorted_obs[o.station_id].append(o)
    ts_of = {s: [o.ts for o in rows] for s, rows in sorted_obs.items()}
    by_ts = {s: {o.ts: o for o in rows} for s, rows in sorted_obs.items()}
    pairs: list[tuple[datetime, str, datetime, float]] = []  # (origin, station, target, y)
    for s in sorted(by_ts):
        for target, label in by_ts[s].items():
            origin = target - horizon * HOUR
            if persistence_value(by_ts[s].get, origin) is None:
                continue  # no persistence value: never scored, never a training pair of the gate
            pairs.append((origin, s, target, label.value))
    pairs.sort(key=lambda p: (p[0], p[1]))
    tgt = np.array([p[2].timestamp() for p in pairs])
    y_all = np.array([p[3] for p in pairs])
    out_y, out_t, out_p = [], [], []
    for tr, val in inner_cv_masks(tgt, horizon, inner_folds):
        if tr.sum() < 100 or val.sum() < 20:
            continue
        inner_refit = datetime.fromtimestamp(float(tgt[tr].max()), UTC) + HOUR
        inner = _sub_train(train, inner_refit, horizon)
        predictors = [member_factory(n)(inner) for n in names]
        idx = np.flatnonzero(val)  # pairs are sorted by origin: chains advance in time order
        cols = np.full((len(idx), len(names)), np.nan)
        for r, i in enumerate(idx):
            origin, s = pairs[i][0], pairs[i][1]
            rows = sorted_obs[s][: bisect_right(ts_of[s], origin - HOUR)]
            snap = Snapshot(origin, s, tuple(o for o in rows if o.available_at <= origin), ())
            for c, predict in enumerate(predictors):
                v = predict(snap, horizon)
                if v is not None and math.isfinite(v):
                    cols[r, c] = max(float(v), 0.0)
        out_y.append(y_all[idx])
        out_t.append(tgt[idx])
        out_p.append(cols)
    result = {
        "y": np.concatenate(out_y) if out_y else np.empty(0),
        "target_s": np.concatenate(out_t) if out_t else np.empty(0),
        "preds": np.vstack(out_p) if out_p else np.empty((0, len(names))),
    }
    _OOF_CACHE[key] = result
    return result


# ---- factory -------------------------------------------------------------------------------


def _write_diagnostics(variant: str, refit: datetime, record: Mapping[str, Any]) -> None:
    path = log_dir() / f"{variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    data[refit.isoformat()] = record  # idempotent: the probe refits the same fold again
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
    spec = {**DEFAULTS, **(params or {})}
    variant = str(spec["variant"])
    if variant not in VARIANT_MEMBERS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANT_MEMBERS)}")
    names = VARIANT_MEMBERS[variant]
    k = len(names)
    # (weights, intercept) per horizon
    weights: dict[int, tuple[np.ndarray, float]] = {}
    record: dict[str, Any] = {"variant": variant, "members": list(names), "horizons": {}}
    for h in train.horizons:
        w, b, info = np.full(k, 1.0 / k), 0.0, {"fallback_equal": False}
        if variant in FITTED_VARIANTS:
            oof = oof_predictions(train, names, h, int(spec["inner_folds"]))
            ok = np.isfinite(oof["preds"]).all(axis=1)
            info["oof_rows"] = int(ok.sum())
            if ok.sum() >= int(spec["min_oof_rows"]):
                y, p = oof["y"][ok], oof["preds"][ok]
                info["oof_mae"] = dict(
                    zip(names, map(float, np.abs(p - y[:, None]).mean(axis=0)), strict=True)
                )
                if variant == "invmae":
                    w, err, used = inverse_mae_weights(
                        oof["y"], oof["preds"], oof["target_s"], float(spec["weights_window_days"])
                    )
                    info["window_rows"] = used
                    info["window_mae"] = dict(zip(names, map(float, err), strict=True))
                else:
                    b, w = fit_stacker(y, p)
                info["oof_mae_ensemble"] = float(np.abs(b + p @ w - y).mean())
            else:
                info["fallback_equal"] = True
        weights[h] = (w, b)
        info.update(weights=dict(zip(names, map(float, w), strict=True)), intercept=float(b))
        record["horizons"][str(h)] = info
    _write_diagnostics(variant, train.refit_time, record)

    predictors = [member_factory(n)(train) for n in names]
    held: list[Snapshot | None] = [None, None]  # (incoming snapshot, its clean copy)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        wb = weights.get(horizon)
        if wb is None:
            return None
        if held[0] is not snapshot:  # several horizons are asked of the same snapshot object
            held[0], held[1] = snapshot, clean_snapshot(snapshot)
        clean = held[1]
        preds = []
        for member in predictors:
            v = member(clean, horizon)
            if v is None or not math.isfinite(v):
                return None
            preds.append(max(float(v), 0.0))
        return combine(preds, wb[0], wb[1])

    return predict
