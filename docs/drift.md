# Drift detection (KAN-22)

`python -m pm25_forecast.drift` runs daily (see `docs/drift-workflow.yml.txt`). It asks one
question: do the model's inputs, and the PM2.5 it is trying to predict, still look like the data
the model was trained on?

## What PSI means in plain words

The Population Stability Index (PSI) measures how much a distribution has moved. We cut the
training-time values into 5 bins that each hold about 20% of that data, then count what share of
the last 7 days falls into each bin. If the shares are still about 20% each, nothing moved and PSI
is near 0. The more the recent shares differ from the training shares, the bigger the PSI. It says
that something changed, not whether the model got worse: scoring (KAN-20) says that.

| PSI | Status | Meaning |
|---|---|---|
| below 0.1 | `ok` | No meaningful change. |
| 0.1 to 0.2 (inclusive) | `warn` | A moderate change. Look at it, no action needed. |
| above 0.2 | `alert` | A large change, **seen on 2 consecutive daily runs**. One alert is sent; KAN-24 (retrain) can read it. |
| above 0.2, first time | `warn` | Could be a one-day blip. The PSI is stored; the status waits for the next run. |
| not computed | `insufficient` | See below. Stored with no PSI, never alerts. |

The thresholds and minimum sizes are the constants `PSI_WARN`, `PSI_ALERT`, `PSI_BINS`,
`MIN_RECENT`, `MIN_REFERENCE`, `MIN_BINS` and `CAMS_MIN_COVERAGE` in `src/pm25_forecast/drift.py`.
An empty bin counts as a 0.0001 share so the logarithm is defined. Non-finite values are dropped.

"Consecutive" means the previous run (within 2 days) of the same model and feature was also above
0.2. A gap longer than that, or a previous run at or below 0.2, restarts the count.

### When a check is `insufficient`

- Fewer than 150 recent points (a full week is 168 per station and horizon) or fewer than 200
  reference points.
- The reference has fewer than 3 distinct bins. Near-constant or integer-heavy data (many equal
  values) collapses the quantile edges, and PSI on 2 bins is too coarse to trust.
- `cams_pm25` only: CAMS must be present for at least 50% of the reference origins. CAMS vintages
  exist only since 2026-10-27, so for a training window before that this feature is always
  `insufficient`.

## What is compared

- Models: the shadow challenger (`CHALLENGER_MODEL_VERSION`) and the production champion, if one
  is registered (checked once if they are the same).
- Reference: the model manifest's `training_window` (start to end of the training origins).
- Recent: the last 7 days up to the current UTC hour.
- Features: every name in `FEATURE_NAMES` (`pm25_latest`, `pm25_same_hour_24h`, `cams_pm25`),
  rebuilt hourly per station from stored observations and CAMS vintages through the shared
  `build_features`, like `predict` does: one feature set per origin and per horizon the model was
  trained for (the manifest `metrics` keys; 24 h when they name none). The observation history
  is the 24 h up to `assumed_reporting_lag_hours` before the origin, as in training. Stations are
  pooled; `STATION_LOCATIONS` limits them when set.
- `target_pm25`: the observed PM2.5 itself, recent 7 days against the training window.

A 24 h-only model with a reporting lag of 1 h or more never sees the reading of the origin hour,
in training or in serving, so `pm25_same_hour_24h` is always empty for it and is `insufficient`.
That is correct, not a gap in the check. Models with shorter horizons (6 h, 12 h) do have it.

## Cost: the reference is cached

The reference side is computed once per `model_version` (a model's training window never
changes) and stored in `drift_reference` as bin edges and shares. A daily run reads only the
recent 7 days from the database. Delete a model's `drift_reference` rows to force a rebuild.

## Where results go

Table `drift_checks` (migration `004_drift_checks.sql`): one row per model and feature (or
`target_pm25`) per UTC day, keyed `(checked_at, model_version, feature)`. `checked_at` is the UTC
day, so a re-run the same day updates the rows (`computed_at` shows the last run).

`latest_drift_alert(con, model_version=None)` is true when any row of the latest run (optionally of
one model) is `alert`. A latest run computed more than 36 h ago counts as no alert, so a stale
alert cannot trigger a retrain again.

One alert per run lists the model/feature pairs in `alert` status. Exit codes: 0 ok (drift alerts
included), 1 failed or database down (run skipped, alert sent), 2 configuration error.
