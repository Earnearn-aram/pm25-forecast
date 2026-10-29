"""Load a registered LightGBM model for hourly serving and predict from shared features (KAN-29).

Two bundle shapes are registered under ``format: 1`` (the label does not tell them apart):

* multi-horizon (KAN-32 ``train_model``): ``{"format": 1, "horizons": {"6": str, ...}}``
* single-horizon (KAN-15 ``retrospective``): ``{"format": 1, "horizon_hours": 24, "booster": str}``

The shape is detected from the keys, and an artifact that matches neither, or both, is rejected.
Every booster's feature names must equal the manifest ``feature_list`` exactly, in order, and every
name must be produced by the shared feature builder. Any mismatch raises ``ModelLoadError`` so the
prediction job fails loudly instead of feeding features in the wrong order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb

from pm25_forecast.features import FEATURE_NAMES
from pm25_forecast.models.baselines import HORIZONS_H
from pm25_forecast.storage import registry


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
        return max(0.0, float(booster.predict(vector)[0]))


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
    boosters = _parse_bundle(artifact)
    for horizon, booster in boosters.items():
        if tuple(booster.feature_name()) != feature_list:
            raise ModelLoadError(
                f"+{horizon}h booster features {booster.feature_name()} "
                f"do not match manifest feature_list {list(feature_list)}"
            )
    return ServingModel(model_version, feature_list, boosters)


def load_registered_model(con: Any, model_version: str) -> ServingModel:
    """Read a content-verified bundle from the registry and prepare it for serving."""
    try:
        artifact, manifest = registry.read_model(con, model_version)
    except LookupError as exc:
        raise ModelLoadError(str(exc)) from None
    except ValueError as exc:
        raise ModelLoadError(f"registry integrity check failed: {exc}") from None
    return load_serving_model(model_version, artifact, manifest)
