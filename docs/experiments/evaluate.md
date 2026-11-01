# Evaluation runner (KAN-82, E04)

`src/pm25_forecast/experiments/evaluate.py` runs the walk-forward evaluation of
`protocol.md` sections 4-6 on a frozen E02 snapshot and logs the run through `tracking.log_run`.
Statistics live in `experiments/stats.py` (numpy and math only, no scipy).

## Run it

    uv sync --locked --group experiments        # once; mlflow is needed for logging
    uv run --group experiments python -m pm25_forecast.experiments.evaluate \
        --prereg docs/experiments/prereg/<id>.yaml \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 \
        --model my_models.lgbm:fit

| Flag | Meaning |
|---|---|
| `--prereg` | pre-registration YAML/JSON. Without it the run is tagged `prereg=exploratory` (never holdout-eligible) and the model name comes from the factory name |
| `--snapshot` | E02 snapshot directory (hash is re-verified on load) |
| `--model module:factory` | the model factory, see below. Built-ins: `pm25_forecast.experiments.evaluate:{persistence,seasonal_naive,smoother}` |
| `--holdout` | (with it, `--horizons` and `--finalists` are rejected: both come from the prereg) score the locked holdout instead of dev. Refused unless the prereg is `final: true`, `split: holdout`, the git tree is clean, logging is on, and no holdout run with this prereg id exists yet |
| `--finalists k` | dev only; default is `prereg.finalists`. CI level is `1 - 0.05/k` (logged as `ci_level`) |
| `--horizons 6 12 24` | dev only; default: the prereg horizons, else 6 12 24 |
| `--no-log` | dry run, nothing written to MLflow (not allowed with `--holdout`) |
| `--experiment`, `--run-name`, `--ticket`, `--agent`, `--out-dir` | overrides; default experiment is `kan78/<prereg id>` or `kan78/e04-<model>` |

Exit code 0 = done, 2 = refused (the reason is logged). The summary is printed; the full result
(per-station and per-hour MAE, per-fold diffs, CI sensitivities, DM variants, spike counts,
validity reasons) is `evaluation.json`, attached to the MLflow run as an artifact.

## Writing a model

The complete, tested example is `src/pm25_forecast/experiments/examples/ridge_linear.py` (ridge
on `[1, last value, mean24]`, numpy only); run it with
`--model pm25_forecast.experiments.examples.ridge_linear:fit`. Copy it as a starting point.

A factory is `fit(train: TrainSnapshot[, params]) -> predict`. Called once per weekly fold.

**What you receive** (all datetimes UTC and timezone-aware):

- `TrainSnapshot(refit_time, stations, horizons, observations, vintages)`.
- `Observation(station_id, ts, value, available_at)`: one hourly PM2.5 reading. Rows in
  `train.observations` all have `ts <= refit_time` and `available_at <= refit_time`.
- `Vintage(station_id, issue_time, valid_time, value, available_at, model)`: a stored CAMS
  forecast (the default snapshot has none).
- `params`: optional second argument, receives `prereg.params` (declare it to get it).

**Training pairs** for horizon `h`: every observation is a label with `target = ob.ts`; its
origin is `target - h hours`. Both are `<= refit_time`, so the pair has matured. For the
features at that origin use only rows with `ts <= origin - 1 h` and `available_at <= origin`
(the serving lag), per station:

```python
from pm25_forecast.features import build_features, mean_24h

window = [by_ts[origin - k * HOUR] for k in range(1, 25) if (origin - k * HOUR) in by_ts]
latest = build_features(window, station_vintages, origin, target)["pm25_latest"]  # may be None
mean24 = mean_24h(window, origin)  # may be None
```

`build_features(observations, vintages, prediction_time, valid_time)` returns
`pm25_latest`, `pm25_same_hour_24h`, `cams_pm25`; it ignores anything later than
`prediction_time`. Never write feature code of your own: use `pm25_forecast.features`.

**The predictor** is the `backtest.Predictor` contract (`docs/backtest.md`):
`predict(snapshot: Snapshot, horizon: int) -> float | None`. `snapshot.prediction_time` is the
origin, `snapshot.station_id` the station, `snapshot.observations` / `snapshot.vintages` only
rows available at the origin (`snapshot.features(valid_time)` calls `build_features`). Return
`None` when an input is missing (never impute); negatives are clipped to 0 and counted; NaN or
inf aborts the run. Use only what is in `train` and `snapshot`; do not read the snapshot
directory or other data yourself.

## What the runner does

- **Data**: air4thai era only (`ts >= S0`), targets before `2026-10-26`; the cohort is the
  stations with >= 80% hourly coverage in `[S0, H0)`. Fold dates come from `folds.derive_folds`.
  A dev run never sees any observation at or after `H0`.
- **Refit**: weekly, at `fold start - max(horizons)` (24 h by default), so even the +24 h
  origins of the first day of a fold are forecast by a model that only knew the past.
- **Baselines** are always computed with the same code on the same rows: persistence (3 h
  limit), the KAN-76 smoother (beta 0.5, `features.mean_24h`) and seasonal-naive (same hour
  one or more whole days before the target, newest with `ts <= origin - 1 h`).
- **Pairing**: the model, persistence and smoother are scored on the common intersection of
  (station, origin, horizon). Seasonal-naive is reported on the subset of those rows where it
  exists (`baselines.seasonal_naive.n_pairs`). Coverage below 95% of baseline-eligible pairs marks the run `invalid`.
- **Inference**: hourly pair-weighted series; 72 h moving-block bootstrap (B 10,000, seed
  20261101) for `diff_vs_persistence` and `diff_vs_smoother`, with 24 h / 168 h sensitivities
  and a `fragile` flag; Diebold-Mariano (absolute error, Newey-West, m = 167, HLN, Student-t)
  against the smoother (hours with no pairs are dropped, not filled), with m = 47 and m = h-1 as sensitivities in the JSON.
- **Other**: spike recall/precision (> 37.5, null below 50 observed spikes), MAE by station and
  by target UTC hour, validity reasons (`invalid` tag when any).

## Promotion-gate checks (KAN-96)

Both are computed on every run and written to `evaluation.json`; the runner never decides promotion.

- **Gate 10.3, station vs persistence** (`scores["24"]["station_gate"]`, +24 h only): per station
  `rel_diff = MAE_model / MAE_persistence - 1` with a 72 h block-bootstrap CI (same blocks, seed and
  level as the pooled comparison; ratio of sums). `status` is `fail` if any station's point estimate is
  strictly above +5%, `insufficient_data` if more than 10% of the cohort has fewer than 250 pairs,
  else `pass`. Stations whose CI contains +5% are in `ci_includes_limit`. Metrics:
  `station_worst_rel_diff_24h`, `station_worst_rel_diff_ci_high_24h`, `stations_worse_than_limit_24h`.
- **Gate 10.4, serving lag** (`serving_lag`): 48 sampled origins per fold are re-run with all rows
  that violate the lag (and vintages issued/available after the origin) added to the snapshot with
  value +1000. Same prediction required (`status: pass`); a change is `fail` (the run is tagged
  `invalid`); a model that raises on the widened input is `inconclusive`. Metrics:
  `serving_lag_checks`, `serving_lag_mismatches`. This does not detect a model reading data from
  outside `train`/`snapshot` (forbidden by contract; review it).

A model identical to a baseline has all-zero paired differences against it: its CI and DM are
omitted (the JSON says so), never logged as 0.

## Worked example: the three baselines (dev split)

    for m in persistence smoother seasonal_naive; do
      uv run --group experiments python -m pm25_forecast.experiments.evaluate \
          --snapshot ~/pm25-experiments/snapshots/e02-v1 \
          --model pm25_forecast.experiments.evaluate:$m
    done
    uv run --group experiments python -m pm25_forecast.experiments.tracking leaderboard --format md

Each run takes a few seconds (the guard: persistence on the full dev split must stay far below
5 minutes). Output of one run:

    split=dev folds=5 stations=5
    +24h n=4200 MAE=4.311 RMSE=5.896 skill=+0.000 vs_pers=+0.000 n/a vs_smoother=+0.444 [+0.295, +0.643]
