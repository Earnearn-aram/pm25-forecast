"""KAN-81: persistence demo scoring (stdlib, no mlflow)."""

import math
from pathlib import Path

from pm25_forecast.experiments.demo import load_series, persistence_scores

FIXTURE = Path(__file__).parent / "fixtures" / "tracking_demo_series.json"


def test_persistence_counts_and_errors():
    out = persistence_scores([1.0, None, 3.0, 5.0], horizon_h=2)
    assert out == {"mae_24h": 2.0, "rmse_24h": 2.0, "n_pairs_24h": 1}
    assert persistence_scores([1.0], horizon_h=2) == {"n_pairs_24h": 0}


def test_fixture_series_has_72_pairs():
    scores = persistence_scores(load_series(FIXTURE))
    assert scores["n_pairs_24h"] == 72
    assert math.isfinite(scores["mae_24h"]) and scores["rmse_24h"] >= scores["mae_24h"]
