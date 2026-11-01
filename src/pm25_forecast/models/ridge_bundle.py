"""Registry bundle for the pooled ridge challenger (KAN-100): parsing and validation only.

Light on purpose (no numpy, no LightGBM, no scikit-learn): the registry, the serving loader and
the training script all import it. Layout, registered under ``format: 1``::

    {"format": 1, "ridge": {
        "feature_set": "fs_v2_core", "features": [...], "stations": [...],
        "target": "raw", "output": "change", "missing_policy": "skip" | "train_mean",
        "horizons": {"24": {"intercept": b, "coef": [...], "mean": [...], "scale": [...],
                            "lambda": l, "n_train": n}}}}

``coef``, ``mean`` and ``scale`` have one entry per feature followed by one per station (the
pooled one-hot columns, in ``stations`` order). The prediction is
``latest + intercept + coef . ((x - mean) / scale)`` (change-from-latest, raw target), where
``latest`` is the newest of ``pm25_lag_1h/2h/3h``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from pm25_forecast.features.library_v2 import ALL_FEATURE_NAMES

RIDGE_KEY = "ridge"
MAX_HORIZON_H = 24
MISSING_POLICIES = ("skip", "train_mean")
SPEC_KEYS = {
    "feature_set",
    "features",
    "stations",
    "target",
    "output",
    "missing_policy",
    "horizons",
}
HORIZON_KEYS = {"intercept", "coef", "mean", "scale", "lambda", "n_train"}


@dataclass(frozen=True)
class RidgeHorizonSpec:
    intercept: float
    coef: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    strength: float
    n_train: int


@dataclass(frozen=True)
class RidgeSpec:
    feature_set: str
    features: tuple[str, ...]
    stations: tuple[str, ...]
    missing_policy: str
    horizons: dict[int, RidgeHorizonSpec]


def is_ridge_bundle(artifact: bytes) -> bool:
    try:
        bundle = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(bundle, dict) and RIDGE_KEY in bundle


def _finite_list(value: Any, size: int, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(isinstance(v, bool) or not isinstance(v, int | float) for v in value)
        or not all(math.isfinite(v) for v in value)
    ):
        raise ValueError(f"ridge bundle {label} must be {size} finite numbers")
    return tuple(float(v) for v in value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"ridge bundle {label} must be a finite number")
    return float(value)


def parse_ridge_bundle(payload: Any) -> RidgeSpec:
    """Validate the ``ridge`` section; raise ``ValueError`` on anything unexpected."""
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("unsupported model bundle format")
    spec = payload.get(RIDGE_KEY)
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError(f"ridge bundle needs exactly the keys {sorted(SPEC_KEYS)}")
    features, stations = spec["features"], spec["stations"]
    for label, names in (("features", features), ("stations", stations)):
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"ridge bundle {label} must be a non-empty list of unique names")
    unknown = [n for n in features if n not in ALL_FEATURE_NAMES]
    if unknown:
        raise ValueError(f"ridge features not produced by the shared builder: {', '.join(unknown)}")
    if spec["target"] != "raw" or spec["output"] != "change":
        raise ValueError("only the raw-target change-from-latest ridge is supported")
    if not isinstance(spec["feature_set"], str) or spec["missing_policy"] not in MISSING_POLICIES:
        raise ValueError(f"ridge bundle missing_policy must be one of {MISSING_POLICIES}")
    horizons = spec["horizons"]
    if not isinstance(horizons, dict) or not horizons:
        raise ValueError("ridge bundle has no horizons")
    size = len(features) + len(stations)
    out: dict[int, RidgeHorizonSpec] = {}
    for key, item in horizons.items():
        try:
            horizon = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"invalid horizon in ridge bundle: {key!r}") from None
        if not 1 <= horizon <= MAX_HORIZON_H or str(horizon) != key:
            raise ValueError(f"ridge horizon +{key}h is outside 1..{MAX_HORIZON_H}")
        if not isinstance(item, dict) or set(item) != HORIZON_KEYS:
            raise ValueError(f"ridge +{horizon}h needs exactly the keys {sorted(HORIZON_KEYS)}")
        scale = _finite_list(item["scale"], size, f"+{horizon}h scale")
        if any(s <= 0 for s in scale):
            raise ValueError(f"ridge +{horizon}h scale must be positive")
        n_train = item["n_train"]
        if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train < 1:
            raise ValueError(f"ridge +{horizon}h n_train must be a positive integer")
        out[horizon] = RidgeHorizonSpec(
            _number(item["intercept"], f"+{horizon}h intercept"),
            _finite_list(item["coef"], size, f"+{horizon}h coef"),
            _finite_list(item["mean"], size, f"+{horizon}h mean"),
            scale,
            _number(item["lambda"], f"+{horizon}h lambda"),
            n_train,
        )
    return RidgeSpec(
        spec["feature_set"],
        tuple(features),
        tuple(stations),
        spec["missing_policy"],
        dict(sorted(out.items())),
    )


def validate_ridge_artifact(artifact: bytes, feature_list: list[str]) -> RidgeSpec:
    """Registry check: parse the bundle and require its features to equal the manifest's."""
    try:
        bundle = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model artifact is not a supported JSON bundle") from exc
    spec = parse_ridge_bundle(bundle)
    if list(spec.features) != feature_list:
        raise ValueError("manifest feature_list does not match model artifact features")
    return spec
