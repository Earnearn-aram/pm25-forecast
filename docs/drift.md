# Drift detection (KAN-22)

`python -m pm25_forecast.drift` runs daily (see `docs/drift-workflow.yml.txt`). It asks one
question: do the model's inputs, and the PM2.5 it is trying to predict, still look like the data
the model was trained on?

## What PSI means in plain words

The Population Stability Index (PSI) measures how much a distribution has moved. We cut the
training-time values into 10 bins that each hold about 10% of that data, then count what share of
the last 7 days falls into each bin. If the shares are still about 10% each, nothing moved and PSI
is near 0. The more the recent shares differ from the training shares, the bigger the PSI. It says
that something changed, not whether the model got worse: scoring (KAN-20) says that.

| PSI | Status | Meaning |
|---|---|---|
| below 0.1 | `ok` | No meaningful change. |
| 0.1 to 0.2 (inclusive) | `warn` | A moderate change. Look at it, no action needed. |
| above 0.2 | `alert` | A large change. One alert is sent; KAN-24 (retrain) can read it. |
| not computed | `insufficient` | Fewer than 48 recent or 200 reference points. Stored with no PSI, never alerts. |

The thresholds and minimum sizes are the constants `PSI_WARN`, `PSI_ALERT`, `MIN_RECENT` and
`MIN_REFERENCE` in `src/pm25_forecast/drift.py`. An empty bin counts as a 0.0001 share so the
logarithm is defined. Non-finite values are dropped.

## What is compared

- Models: the shadow challenger (`CHALLENGER_MODEL_VERSION`) and the production champion, if one
  is registered (checked once if they are the same).
- Reference: the model manifest's `training_window` (start to end of the training origins).
- Recent: the last 7 days up to the current UTC hour.
- Features: every name in `FEATURE_NAMES` (`pm25_latest`, `pm25_same_hour_24h`, `cams_pm25`),
  rebuilt hourly per station from stored observations and CAMS vintages through the shared
  `build_features` (24 h ahead, observations at least `assumed_reporting_lag_hours` old, as in
  training). Stations are pooled; `STATION_LOCATIONS` limits them when set.
- `target_pm25`: the observed PM2.5 itself, recent 7 days against the training window.

Limits to know: CAMS vintages exist only since 2026-10-27, so `cams_pm25` is `insufficient` for
a training window before that. With a reporting lag of 1 h or more `pm25_same_hour_24h` is not
available (the origin's own reading is not yet reported), so it is `insufficient` too.

## Where results go

Table `drift_checks` (migration `004_drift_checks.sql`): one row per model and feature (or
`target_pm25`) per UTC day, keyed `(checked_at, model_version, feature)`. `checked_at` is the UTC
day, so a re-run the same day updates the rows (`computed_at` shows the last run).
`latest_drift_alert(con) -> bool` is true when any row of the latest day is `alert`.

One alert per run lists the model/feature pairs in `alert` status. Exit codes: 0 ok (drift alerts
included), 1 failed or database down (run skipped, alert sent), 2 configuration error.
