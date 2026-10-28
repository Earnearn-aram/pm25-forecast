"""Synthetic-only rolling-origin and future-vintage regression tests."""

from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast.backtest import Observation, Snapshot, Vintage, evaluate

START = datetime(2026, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)


def run(obs=(), vintages=(), **kw):
    return evaluate(
        obs,
        vintages,
        **{
            "start": START,
            "end": START + H,
            "as_of": START + 48 * H,
            "stations": ["synthetic"],
            "aqi_breakpoints": [15, 25, 37.5, 75],
            **kw,
        },
    )


def observation(hour, value, available=None):
    ts = START + hour * H
    return Observation("synthetic", ts, value, ts if available is None else available)


def vintage(issue, target, value, available=None):
    ts = START + issue * H
    return Vintage(
        "synthetic", ts, START + target * H, value, ts if available is None else available
    )


def test_identical_pairs_metrics_and_station_month():
    report = run(
        [observation(0, 10), observation(-18, 20), observation(6, 30), observation(12, 90)],
        [vintage(-1, 6, 25)],
    )
    # +12h actual exists but CAMS/seasonal missing: excluded from EVERY model.
    assert report["paired_rows"] == 1
    assert report["missing"]["cams"] == 1
    metrics = report["metrics_by_horizon"]["6"]
    assert metrics["cams"] == {"n": 1, "mae": 5, "rmse": 5, "aqi_category_accuracy": 0}
    assert metrics["persistence-24h"]["mae"] == 20
    assert metrics["seasonal-naive"]["mae"] == 10
    assert report["mae_by_station_month_horizon"]["synthetic/2026-01/+6h"]["cams"] == 5
    assert report["metrics_by_horizon"]["12"]["cams"]["mae"] is None


def test_future_features_vintages_and_unavailable_observations_never_reach_model():
    def challenger(snapshot, horizon):
        assert all(o.ts <= snapshot.prediction_time for o in snapshot.observations)
        assert all(o.available_at <= snapshot.prediction_time for o in snapshot.observations)
        assert all(v.issue_time <= snapshot.prediction_time for v in snapshot.vintages)
        assert all(v.available_at <= snapshot.prediction_time for v in snapshot.vintages)
        assert {o.value for o in snapshot.observations} == {10}
        assert {v.value for v in snapshot.vintages} == {12}
        return 10

    report = run(
        [
            observation(0, 10),
            observation(1, 999),
            observation(-1, 999, START + H),
            observation(24, 15),
        ],
        [vintage(0, 24, 12), vintage(1, 24, 999), vintage(-1, 24, 999, START + H)],
        challengers={"challenger": challenger},
        horizons=[24],
    )
    assert report["metrics_by_horizon"]["24"]["cams"]["mae"] == 3
    assert report["beats_cams"]["challenger"]["status"] == "insufficient_data"
    with pytest.raises(ValueError, match="feature/vintage"):
        Snapshot(START, "synthetic", (), (vintage(1, 24, 999),))
    with pytest.raises(ValueError, match="observation/label"):
        Snapshot(START, "synthetic", (observation(1, 999),), ())


def test_rolling_fit_sees_only_expanding_mature_history():
    seen = []

    def fit(snapshot, horizon):
        seen.append((snapshot.prediction_time, len(snapshot.observations)))
        return 10

    run(
        [observation(n, 10) for n in range(30)],
        [vintage(n, n + 24, 10) for n in range(2)],
        end=START + 2 * H,
        horizons=[24],
        challengers={"fitted": fit},
    )
    assert seen == [(START, 1), (START + H, 2)]


def test_missing_archive_never_fabricates_cams_or_a_win():
    report = run([observation(n, 10) for n in range(49)], challengers={"x": lambda *_: 10})
    assert report["status"] == "insufficient_data"
    assert report["paired_rows"] == 0
    assert report["beats_cams"]["x"]["beats_cams"] is None


def test_paired_daily_uncertainty_requires_28_days_and_fixed_station_coverage():
    obs = [observation(n, 10) for n in range(28 * 24 + 24)]
    cams = [vintage(n, n + 24, 20) for n in range(28 * 24)]
    options = {
        "end": START + timedelta(days=28),
        "as_of": START + timedelta(days=30),
        "horizons": [24],
        "challengers": {"better": lambda *_: 10, "tie": lambda *_: 20},
    }
    report = run(obs, cams, **options)
    assert report["beats_cams"]["better"]["beats_cams"] is True
    assert report["beats_cams"]["better"]["upper_95"] == -10
    assert report["beats_cams"]["tie"]["beats_cams"] is False
    # Losing a whole origin day must never be turned into a significant win.
    incomplete = run(obs, cams[24:], **options)
    assert incomplete["beats_cams"]["better"]["status"] == "insufficient_data"
    assert incomplete["beats_cams"]["better"]["beats_cams"] is None
    assert (
        run(obs, cams, **{**options, "stations": ["synthetic", "missing"]})["beats_cams"]["better"][
            "status"
        ]
        == "insufficient_data"
    )


def test_duplicates_are_rejected_to_preserve_first_forecast_vintage():
    row = vintage(0, 24, 10)
    with pytest.raises(ValueError, match="first write"):
        run(vintages=[row, row])
    row = observation(0, 10)
    with pytest.raises(ValueError, match="revision history"):
        run([row, row])


@pytest.mark.parametrize(
    "change",
    [
        {"start": START.replace(tzinfo=None)},
        {"end": START},
        {"horizons": [0]},
        {"stations": []},
        {"aqi_breakpoints": [25, 15]},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        run(**change)


def test_invalid_values_and_naive_input_are_rejected():
    with pytest.raises(ValueError):
        run([observation(0, float("nan"))])
    with pytest.raises(ValueError):
        run([Observation("synthetic", START.replace(tzinfo=None), 10, START)])


def test_unmatured_or_late_actuals_not_scored():
    report = run(
        [observation(0, 10), observation(24, 10, START + 100 * H)],
        [vintage(0, 24, 10)],
        horizons=[24],
    )
    assert report["eligible_actual_pairs"] == 0


def test_stored_challenger_uses_available_archive_only():
    from pm25_forecast.backtest import stored_predictor

    snapshot = Snapshot(
        START,
        "synthetic",
        (),
        (
            Vintage("synthetic", START - H, START + 24 * H, 15, START - H, "candidate"),
            Vintage("synthetic", START, START + 24 * H, 12, START, "candidate"),
        ),
    )
    assert stored_predictor("candidate")(snapshot, 24) == 12
    assert stored_predictor("missing")(snapshot, 24) is None
    with pytest.raises(ValueError):
        run(horizons=[24, 24])


def test_cli_reports_insufficient_data_without_inventing_values(tmp_path, monkeypatch, capsys):
    import json

    from pm25_forecast import backtest

    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "start": START.isoformat(),
                "end": (START + H).isoformat(),
                "as_of": (START + 48 * H).isoformat(),
                "stations": ["synthetic"],
                "aqi_breakpoints": [15, 25],
                "observations": [
                    {
                        "station_id": "synthetic",
                        "ts": START.isoformat(),
                        "value": 10,
                        "available_at": START.isoformat(),
                    }
                ],
                "vintages": [],
                "challenger_models": ["candidate"],
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["backtest", str(path)])
    assert backtest.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "insufficient_data"
    assert report["beats_cams"]["candidate"]["beats_cams"] is None
