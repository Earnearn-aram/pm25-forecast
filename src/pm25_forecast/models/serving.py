"""Load a registered LightGBM model for hourly serving and predict from shared features (KAN-29).

Two bundle shapes are registered under ``format: 1`` (the label does not tell them apart):

* multi-horizon (KAN-32 ``train_model``): ``{"format": 1, "horizons": {"6": str, ...}}``
* single-horizon (KAN-15 ``retrospective``): ``{"format": 1, "horizon_hours": 24, "booster": str}``

Reporting lag: the hourly job builds features as of the floored issue hour, so the reading for
that hour has never been ingested yet and the freshest input is at least 1 h old. A manifest that
declares ``assumed_reporting_lag_hours`` below ``MIN_SERVING_LAG_H`` was trained on inputs the live
job never has (train/serve skew) and is rejected. Manifests without the field (KAN-32, trained
with the strict as-of availability gate) are not affected.

The shape is detected from the keys, and an artifact that matches neither, or both, is rejected.
Every booster's feature names must equal the manifest ``feature_list`` exactly, in order, and every
name must be produced by the shared feature builder. Any mismatch raises ``ModelLoadError`` so the
prediction job fails loudly instead of feeding features in the wrong order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np

from pm25_forecast.features import (
    FEATURE_NAMES,
    MEAN_24H_NAME,
    SMOOTHER_FEATURE_NAMES,
    _rows,
)
from pm25_forecast.features.library_v2 import build_features_v2, prepare_series
from pm25_forecast.models.baselines import HORIZONS_H, require_utc
from pm25_forecast.models.ridge_bundle import RidgeSpec, is_ridge_bundle, parse_ridge_bundle
from pm25_forecast.storage import registry

MIN_SERVING_LAG_H = 1
RIDGE_LATEST_LAGS = ("pm25_lag_1h", "pm25_lag_2h", "pm25_lag_3h")  # newest of these = "latest"
RIDGE_LOOKBACK = timedelta(hours=80)  # fs_v2_* read at most 72 h before the origin
HOUR = timedelta(hours=1)


class ModelLoadError(ValueError):
    """The registered model cannot be served safely (bad bundle or feature-spec mismatch)."""


@dataclass(frozen=True)
class ServingModel:
    model_version: str
    feature_list: tuple[str, ...]
    boosters: Mapping[int, lgb.Booster]  # horizon hours -> booster

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.boosters))

    def predict(self, horizon: int, features: Mapping[str, float | None]) -> float | None:
        """Predict one value; None when every model input is missing (nothing to predict from)."""
        booster = self.boosters[horizon]
        missing = [name for name in self.feature_list if name not in features]
        if missing:
            raise ModelLoadError(f"feature builder did not produce: {', '.join(missing)}")
        row = [features[name] for name in self.feature_list]
        if all(value is None for value in row):
            return None
        vector = [[math.nan if value is None else float(value) for value in row]]
        pred = float(booster.predict(vector)[0])
        if not math.isfinite(pred):
            raise ValueError(f"{self.model_version} +{horizon}h produced a non-finite prediction")
        return max(0.0, pred)


@dataclass(frozen=True)
class SmootherModel:
    """``latest + beta * (mean24 - latest)`` (KAN-76); same serving surface as ``ServingModel``."""

    model_version: str
    beta: float
    served_horizons: tuple[int, ...]
    feature_list: tuple[str, ...] = SMOOTHER_FEATURE_NAMES

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.served_horizons

    def predict(self, horizon: int, features: Mapping[str, float | None]) -> float | None:
        """None when there is no latest reading; a missing 24 h mean falls back to the latest."""
        missing = [name for name in self.feature_list if name not in features]
        if missing:
            raise ModelLoadError(f"feature builder did not produce: {', '.join(missing)}")
        latest, mean = features["pm25_latest"], features[MEAN_24H_NAME]
        if latest is None:
            return None
        pred = float(latest) if mean is None else latest + self.beta * (mean - latest)
        if not math.isfinite(pred):
            raise ValueError(f"{self.model_version} +{horizon}h produced a non-finite prediction")
        return max(0.0, pred)


@dataclass(frozen=True, eq=False)
class RidgeModel:
    """Pooled ridge, raw target, change-from-latest (KAN-100). One small ridge per horizon.

    ``predict`` takes the shared feature mapping (``library_v2.build_features_v2`` output, which
    must also hold ``RIDGE_LATEST_LAGS``) plus the station; ``forecast`` builds the features itself
    from raw observations. A missing input gives None for that (station, horizon): with
    ``missing_policy`` ``skip`` any missing feature, with ``train_mean`` (the experiment's rule,
    declared in the bundle) only a missing latest reading. Nothing else is ever filled in.
    """

    model_version: str
    spec: RidgeSpec

    @property
    def feature_list(self) -> tuple[str, ...]:
        return self.spec.features

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(self.spec.horizons)

    @property
    def stations(self) -> tuple[str, ...]:
        return self.spec.stations

    def predict(
        self, horizon: int, features: Mapping[str, float | None], station_id: str
    ) -> float | None:
        fit = self.spec.horizons[horizon]
        need = (*RIDGE_LATEST_LAGS, *self.feature_list)
        missing = [name for name in need if name not in features]
        if missing:
            raise ModelLoadError(f"feature builder did not produce: {', '.join(missing)}")
        if station_id not in self.spec.stations:
            return None  # the model was not fitted for this station
        latest = next((features[n] for n in RIDGE_LATEST_LAGS if features[n] is not None), None)
        if latest is None:
            return None
        row = [features[name] for name in self.feature_list]
        if self.spec.missing_policy == "skip" and any(v is None for v in row):
            return None
        onehot = [float(s == station_id) for s in self.spec.stations]
        x = np.array([np.nan if v is None else float(v) for v in row] + onehot)
        mean, scale = np.array(fit.mean), np.array(fit.scale)
        z = (np.where(np.isnan(x), mean, x) - mean) / scale
        pred = float(latest) + fit.intercept + float(z @ np.array(fit.coef))
        if not math.isfinite(pred):
            raise ValueError(f"{self.model_version} +{horizon}h produced a non-finite prediction")
        return max(0.0, pred)

    def forecast(
        self,
        observations: Iterable[Any],
        station_id: str,
        origin: datetime,
        horizons: Sequence[int] | None = None,
    ) -> dict[int, float | None]:
        """Forecast ``origin + h`` for each served horizon from one station's raw observations.

        Only readings with ``origin - 80 h <= ts <= origin - 1 h`` and ``available_at <= origin``
        are used (the serving lag); anything newer is ignored, whatever the caller passes.
        """
        origin = require_utc(origin)
        fields = (("ts",), ("pm25", "value"), ("available_at", "ingested_at"))
        recent = []
        for row in _rows(observations, fields):
            ts, available_at = row["ts"], row["available_at"]
            if ts is None:
                continue
            ts = require_utc(ts)
            if not origin - RIDGE_LOOKBACK <= ts <= origin - HOUR:
                continue
            if available_at is not None and require_utc(available_at) > origin:
                continue
            recent.append((ts, row["pm25"], available_at))
        prep = prepare_series({station_id: recent})
        names = tuple(dict.fromkeys((*RIDGE_LATEST_LAGS, *self.feature_list)))
        out: dict[int, float | None] = {}
        for h in self.horizons if horizons is None else horizons:
            feats = build_features_v2(
                prep, station_id, origin, origin + h * HOUR, None, names=names
            )
            out[h] = self.predict(h, feats, station_id)
        return out


def parse_smoother_bundle(payload: Any) -> tuple[float, tuple[int, ...]]:
    """Validate the ``smoother`` section of a bundle: beta in [0, 1] and served horizons."""
    spec = payload.get("smoother") if isinstance(payload, dict) else None
    if not isinstance(spec, dict) or set(spec) != {"beta", "horizons"}:
        raise ModelLoadError("smoother bundle needs exactly beta and horizons")
    beta, horizons = spec["beta"], spec["horizons"]
    if isinstance(beta, bool) or not isinstance(beta, int | float) or not 0 <= beta <= 1:
        raise ModelLoadError("smoother beta must be a number in [0, 1]")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(isinstance(h, bool) or h not in HORIZONS_H for h in horizons)
    ):
        raise ModelLoadError(f"smoother horizons must be a non-empty subset of {HORIZONS_H}")
    return float(beta), tuple(sorted(set(horizons)))


def _booster(model_text: Any, label: str) -> lgb.Booster:
    if not isinstance(model_text, str) or not model_text:
        raise ModelLoadError(f"{label}: booster text is missing")
    try:
        return lgb.Booster(model_str=model_text)
    except (lgb.basic.LightGBMError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"{label}: invalid LightGBM booster") from exc


def _parse_bundle(artifact: bytes) -> dict[int, lgb.Booster]:
    try:
        payload = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadError("model artifact is not a JSON bundle") from exc
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ModelLoadError("unsupported model bundle format")
    multi = "horizons" in payload
    single = "horizon_hours" in payload or "booster" in payload
    if multi == single:
        raise ModelLoadError("ambiguous model bundle: expected either horizons or horizon_hours")
    if single and not ("horizon_hours" in payload and "booster" in payload):
        raise ModelLoadError("single-horizon bundle needs both horizon_hours and booster")
    if multi:
        horizons = payload["horizons"]
        if not isinstance(horizons, dict) or not horizons:
            raise ModelLoadError("multi-horizon bundle has no horizons")
        items = list(horizons.items())
    else:
        items = [(payload.get("horizon_hours"), payload.get("booster"))]
    boosters: dict[int, lgb.Booster] = {}
    for key, model_text in items:
        try:
            horizon = int(key)
        except (TypeError, ValueError):
            raise ModelLoadError(f"invalid horizon in bundle: {key!r}") from None
        if horizon not in HORIZONS_H:
            raise ModelLoadError(f"bundle horizon +{horizon}h is not served ({HORIZONS_H})")
        boosters[horizon] = _booster(model_text, f"+{horizon}h")
    return boosters


def load_serving_model(
    model_version: str, artifact: bytes, manifest: Mapping[str, Any]
) -> ServingModel:
    """Build a ServingModel from a verified registry bundle; raise on any spec mismatch."""
    feature_list = tuple(manifest.get("feature_list") or ())
    if not feature_list:
        raise ModelLoadError("manifest feature_list is missing")
    unknown = [name for name in feature_list if name not in FEATURE_NAMES]
    if unknown:
        raise ModelLoadError(
            f"manifest features not produced by the shared feature builder: {', '.join(unknown)}"
        )
    _check_lag(manifest)
    boosters = _parse_bundle(artifact)
    for horizon, booster in boosters.items():
        if tuple(booster.feature_name()) != feature_list:
            raise ModelLoadError(
                f"+{horizon}h booster features {booster.feature_name()} "
                f"do not match manifest feature_list {list(feature_list)}"
            )
    return ServingModel(model_version, feature_list, boosters)


def _check_lag(manifest: Mapping[str, Any]) -> None:
    lag = manifest.get("assumed_reporting_lag_hours")
    if lag is not None and (not isinstance(lag, int | float) or lag < MIN_SERVING_LAG_H):
        raise ModelLoadError(
            f"model was trained with assumed_reporting_lag_hours={lag!r}; the hourly job serves "
            f"inputs at least {MIN_SERVING_LAG_H}h old (retrain with --lag-hours "
            f"{MIN_SERVING_LAG_H})"
        )


def load_ridge_model(
    model_version: str, artifact: bytes, manifest: Mapping[str, Any]
) -> RidgeModel:
    """Build a RidgeModel from a verified registry bundle; raise on any spec mismatch."""
    _check_lag(manifest)
    try:
        spec = parse_ridge_bundle(json.loads(artifact))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelLoadError(f"invalid ridge bundle: {exc}") from None
    if list(spec.features) != list(manifest.get("feature_list") or ()):
        raise ModelLoadError("ridge bundle features do not match manifest feature_list")
    return RidgeModel(model_version, spec)


def load_registered_model(
    con: Any, model_version: str
) -> ServingModel | SmootherModel | RidgeModel:
    """Read a content-verified bundle from the registry and prepare it for serving."""
    try:
        artifact, manifest = registry.read_model(con, model_version)
    except LookupError as exc:
        raise ModelLoadError(str(exc)) from None
    except ValueError as exc:
        raise ModelLoadError(f"registry integrity check failed: {exc}") from None
    if registry.is_smoother_bundle(artifact):
        beta, horizons = parse_smoother_bundle(json.loads(artifact))
        return SmootherModel(model_version, beta, horizons)
    if is_ridge_bundle(artifact):
        return load_ridge_model(model_version, artifact, manifest)
    return load_serving_model(model_version, artifact, manifest)
