"""Persistence +24 h baseline on a recorded hourly series (KAN-81 demo run). Stdlib only."""

import json
import math
from pathlib import Path


def persistence_scores(values: list[float | None], horizon_h: int = 24) -> dict[str, float]:
    """Predict value[t + h] with value[t]; score every pair where both are observed."""
    errors = [
        values[i + horizon_h] - values[i]
        for i in range(len(values) - horizon_h)
        if values[i] is not None and values[i + horizon_h] is not None
    ]
    if not errors:
        return {"n_pairs_24h": 0}
    return {
        "mae_24h": sum(abs(e) for e in errors) / len(errors),
        "rmse_24h": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "n_pairs_24h": len(errors),
    }


def load_series(path: Path | str) -> list[float | None]:
    return json.loads(Path(path).read_text())["values"]
