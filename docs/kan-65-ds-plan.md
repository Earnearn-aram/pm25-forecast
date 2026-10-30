# KAN-65 plan: can we beat persistence at +24 h? (pre-registration)

`[Data Scientist - Sonnet 5.5]`

Written and committed BEFORE any modelling result or EDA statistic was computed. The only things
looked at before this file: table schema, row counts (observations 64,986; forecast_vintages 1,032),
the snapshot SHA-256, and the existing KAN-15/KAN-64 docs and code. Nothing in this plan may be
changed after results; any deviation goes into the report under "Deviations from plan".

## Data

- `data/history.duckdb`, read-only, local export. SHA-256 computed =
  `fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09`, identical to the KAN-15
  snapshot. No database, secret or workflow is touched.
- Table `observations(station_id, ts, pm25, source, available_at)`; five stations (02t, 59t, 61t,
  bkp100t, bkp101t). `forecast_vintages` (CAMS) has only 1,032 rows and is not used.
- Weather history is not in the export: any weather feature is marked "needs data" and not used.
- Known gap: 2026-05-19..21 at bkp100t/bkp101t. Samples are excluded, never imputed. Rows with
  missing target or missing persistence input are dropped (same rule as KAN-15).

## Questions

1. EDA (descriptive, no test): distributions, missingness, diurnal/weekly cycle, autocorrelation at
   1/6/12/24/48/168 h, persistence error by target hour, season, station, PM2.5 level; cross-station
   correlation and lead-lag.
2. Primary: does any model that is servable at a 1 h reporting lag beat persistence on +24 h MAE on
   the KAN-15 test span?
3. Secondary: which feature group (if any) drives a gain; is the residual target better than the
   direct target; is a ridge baseline competitive; is there bias, especially at high PM2.5?

## Hypotheses (directional, with prior expectations)

- H1 Diurnal structure: persistence error depends on hour of day (traffic/boundary-layer cycle), so
  target-hour features lower MAE. Prior: small gain at most; the 24 h horizon is a full cycle so the
  latest value already carries the phase, but not the hour-specific amplitude.
- H2 Residual target (y minus persistence) is easier to learn than y and beats direct y in
  validation MAE.
- H3 Neighbor stations: a network mean of the latest reading reduces noise in the local latest reading
  and helps. Prior: weak, because stations are in one metro area and at 24 h the shared signal is
  mostly the regional level the local latest already holds.
- H4 Smoothing (rolling mean 3/6/24 h) is a better level estimate than a single noisy reading.
- Null expectation: no reliable gain. This is acceptable and will be reported as such.

## Metric and decision rule (fixed now)

- Primary metric: MAE (ug/m3) of the +24 h forecast, pooled station-hours.
- Paired difference d = MAE(candidate) - MAE(persistence) on the identical station-hours, with
  95% percentile CI from a day-block bootstrap (resample target UTC dates with replacement, 10,000
  draws, seed 20261030; same scheme as the KAN-64 review). Sensitivity: 7-day blocks.
- Same comparison against the current LightGBM (`pm25_latest` and `pm25_same_hour_24h`, retrained
  by the same code, direct target, params of `retrospective.py`: regression objective, 200 rounds,
  seed 20261029).
- Persistence = latest reading with ts in [origin-3h, origin-1h] (KAN-15 rule at lag 1).
- PROMOTE only if for the single val-selected primary candidate: (a) 95% CI of d on test is entirely
  below zero AND (b) in both high-pollution subsets (actual target > 37.5 and, as the
  forecast-time-observable version, latest reading at origin > 37.5) the point estimate of d is <= 0.
  Subset CIs are reported; the subset sizes are disclosed because the test span is the wet season.
- Otherwise the answer is "no reliable gain" (or "more data needed" if the CI is too wide to exclude
  a practically relevant gain of 0.25 ug/m3). No test-set tuning.
- Secondary metrics (RMSE, bias, band accuracy) are descriptive only.

## Split (chronological, no shuffles)

- Test: origins >= 2026-07-10 10:00 UTC (KAN-15 `test_start`, 85% point of the origin range) to the
  last origin; identical pair set to KAN-64 (expected 10,729 pairs, 91 target days).
- Train: origins whose target is strictly before the test start (as KAN-15).
- Validation slice inside train for tuning: the last 60 days of training origins. Models used in
  tuning are fit on origins whose target is before the validation start (24 h embargo) and scored on
  validation origins. The final model is then refit on all train rows with the selected config and
  evaluated once on test.
- Rolling-window feature statistics, scalers, imputers are fit on train only.

## As-of rule for features

Reference time r = origin - 1 h. Only readings with ts <= r are used. A "same-hour" lag refers to
readings at target-48 h, target-72 h, target-168 h (all <= origin - 24 h, so always past). Windows
end at r. Calendar features use the target timestamp (known at issue time). Neighbor features use
the other stations' readings with ts in [origin-3h, origin-1h] only. Nothing is forward-filled or
imputed in samples; missing inputs are NaN (LightGBM native; for ridge a train-median fill plus a
missing indicator, fit on train only).

## Feature groups (each tied to an EDA question; groups are cumulative)

| Set | Contents | Motivated by | Servable now |
|---|---|---|---|
| A | latest | current model | yes |
| B | A + sin/cos target hour, sin/cos day-of-week, sin/cos month, station id | H1, season | yes |
| C | B + pm25 at target-48 h, -72 h, -168 h | autocorrelation at 24/48/168 h | yes (needs 7 d of history at predict time) |
| D | C + rolling mean/max/std over 3/6/24 h ending at r, trend = latest minus reading at r-3 h / r-6 h | H4, short-term dynamics | yes |
| E | D + other-station mean of latest, other-station mean of 24 h rolling mean | H3 | yes (needs all 5 stations at predict time) |

Weather, CAMS, wind: "needs data", not used.

## Models and search (predefined, small)

Targets: direct y, or residual y - persistence (prediction = persistence + model output).

- LightGBM, one grid per (feature set, target): objective {l2, l1} x num_leaves {7, 31} x
  learning_rate {0.03, 0.1} x rounds {200, 500}; min_child_samples 50, feature_fraction 0.9,
  seed 20261029, one thread. 16 configs x 5 sets x 2 targets = 160 LightGBM fits.
- Ridge (numpy closed form, standardized, alpha {0.1, 1, 10, 100}) per (set, target): 40 fits.
- Total tuned configurations: 200. Selection metric: validation MAE only.
- Quantile / bias check: the l1 objective is the conditional median; additionally report mean signed
  error and by-level bias of the primary on test.

## Multiplicity

- Exactly ONE primary test comparison (the model with the lowest validation MAE across all 200
  configs), judged at 95%.
- Secondary test comparisons (the best validation config for each of the 5 sets x 2 targets for
  LightGBM, and for each of the 5 sets x 2 targets for ridge = 20, plus the current LightGBM) are
  descriptive; reported with Bonferroni-adjusted CIs (level 1 - 0.05/21). They cannot trigger a
  promotion.

## Reproducibility

Scripts under `src/pm25_forecast/experiments/` (not imported by serving code), run with
`PYTHONPATH=src` from the repo venv. Fixed seeds as above. All report numbers are produced by those
scripts into `out/kan65/*.json`; the report quotes them.
