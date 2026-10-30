"""Target transforms, ridge and LightGBM helpers for the KAN-65 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .common import MODEL_SEED

FIT_COUNT = {"lgbm": 0, "ridge": 0, "lgbm_refit": 0}


def forward(kind: str, y: np.ndarray, latest: np.ndarray) -> np.ndarray:
    if kind == "direct":
        return y
    if kind == "resid":
        return y - latest
    return np.log1p(y) - np.log1p(latest)  # logratio


def inverse(kind: str, out: np.ndarray, latest: np.ndarray) -> np.ndarray:
    if kind == "direct":
        pred = out
    elif kind == "resid":
        pred = latest + out
    else:
        pred = np.expm1(np.log1p(latest) + out)
    return np.maximum(0.0, pred)


def lgb_params(objective: str, leaves: int) -> dict:
    return {
        "objective": objective, "metric": "l1", "learning_rate": 0.05, "num_leaves": leaves,
        "min_child_samples": 50, "feature_fraction": 0.9, "seed": MODEL_SEED,
        "feature_fraction_seed": MODEL_SEED, "bagging_seed": MODEL_SEED, "deterministic": True,
        "force_row_wise": True, "verbosity": -1, "num_threads": 8,
    }


def lgb_fit_es(params, x, y, xv, yv, names, cat, cap=400):
    """Fit with early stopping on the validation slice; returns (booster, best_iter)."""
    FIT_COUNT["lgbm"] += 1
    ds = lgb.Dataset(x, label=y, feature_name=names, categorical_feature=cat, free_raw_data=False)
    dv = lgb.Dataset(xv, label=yv, reference=ds)
    b = lgb.train(params, ds, num_boost_round=cap, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    return b, max(20, b.best_iteration)


def lgb_refit(params, x, y, names, cat, rounds):
    FIT_COUNT["lgbm_refit"] += 1
    ds = lgb.Dataset(x, label=y, feature_name=names, categorical_feature=cat)
    return lgb.train(params, ds, num_boost_round=rounds)


@dataclass
class Ridge:
    names: list
    mu: np.ndarray
    sd: np.ndarray
    miss_cols: np.ndarray
    n_station: int
    w: np.ndarray

    def _design(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for j, n in enumerate(self.names):
            if n == "station_id":
                cols.append(np.eye(self.n_station)[x[:, j].astype(int)])
                continue
            v = x[:, j]
            miss = np.isnan(v)
            z = np.where(miss, 0.0, (v - self.mu[j]) / self.sd[j])
            cols.append(z[:, None])
            if self.miss_cols[j]:
                cols.append(miss.astype(float)[:, None])
        return np.hstack([np.ones((len(x), 1))] + cols)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._design(x) @ self.w


def ridge_fit(x: np.ndarray, y: np.ndarray, names: list, alpha: float, n_station: int = 5) -> Ridge:
    FIT_COUNT["ridge"] += 1
    with np.errstate(all="ignore"):
        mu = np.nanmean(x, 0)
        sd = np.nanstd(x, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    miss_cols = np.isnan(x).any(0)
    r = Ridge(names, mu, sd, miss_cols, n_station, np.zeros(1))
    d = r._design(x)
    pen = alpha * np.eye(d.shape[1])
    pen[0, 0] = 0.0
    r.w = np.linalg.solve(d.T @ d + pen * len(x) / 1000.0, d.T @ y)
    return r
