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
| `--holdout` | score the locked holdout instead of dev. Refused unless the prereg is `final: true`, `split: holdout`, the git tree is clean, logging is on, and no holdout run with this prereg id exists yet |
| `--finalists k` | number of registered finalists; CI level is `1 - 0.05/k` (logged as `ci_level`) |
| `--horizons 6 12 24` | default: the prereg horizons, else 6 12 24 |
| `--no-log` | dry run, nothing written to MLflow (not allowed with `--holdout`) |
| `--experiment`, `--run-name`, `--ticket`, `--agent`, `--out-dir` | overrides; default experiment is `kan78/<prereg id>` or `kan78/e04-<model>` |

Exit code 0 = done, 2 = refused (the reason is logged). The summary is printed; the full result
(per-station and per-hour MAE, per-fold diffs, CI sensitivities, DM variants, spike counts,
validity reasons) is `evaluation.json`, attached to the MLflow run as an artifact.

## Writing a model

```python
# my_models/lgbm.py  (importable from the working directory or installed)
from pm25_forecast.experiments.evaluate import TrainSnapshot

def fit(train: TrainSnapshot, params: dict):      # `params` is optional: it receives prereg.params
    # train.observations / train.vintages: only rows with ts/issue_time <= train.refit_time and
    # available_at <= train.refit_time. Build (origin, origin+h) pairs from them with
    # pm25_forecast.features (do not write feature code of your own) and require origin + h <= refit_time.
    model = ...  # fit here
    def predict(snapshot, horizon: int) -> float | None:
        feats = snapshot.features(snapshot.prediction_time + timedelta(hours=horizon))
        return None if all(v is None for v in feats.values()) else float(model.predict(feats))
    return predict
```

`predict` is the `backtest.Predictor` contract (`docs/backtest.md`): `snapshot` holds only
observations with `ts <= origin - 1 h` and `available_at <= origin`. Return `None` when there
is nothing to predict from (never impute); negatives are clipped to 0 and counted; NaN/inf
aborts the run. The factory is called once per weekly fold.

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
  against the smoother, with m = 47 and m = h-1 as sensitivities in the JSON.
- **Other**: spike recall/precision (> 37.5, null below 50 observed spikes), MAE by station and
  by target UTC hour, validity reasons (`invalid` tag when any).

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
