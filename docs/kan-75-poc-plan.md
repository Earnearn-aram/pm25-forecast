# KAN-75 PoC plan (pre-registered; written and committed before any model was fitted or scored)

Time box: half a day. Offline except public Open-Meteo calls. Not part of the serving path.

## Question and hypothesis

Q: do lead-matched +24 h weather forecasts make the Bangkok PM2.5 +24 h forecast clearly better
than persistence and than the one-parameter smoother (KAN-65 headline simple candidate)?
H1: LightGBM on PM2.5 features + weather forecast features has lower MAE than the smoother.
Weather is expected to help most for the level shift (wind, rain, humidity, boundary layer) that
PM2.5 history cannot see.

## Data

- `data/history.duckdb` (read-only), `source = 'air4thai'` rows only, ts from 2026-07-31 17:00 UTC.
- KAN-74 exclusion: all features and targets strictly before 2026-10-26 00:00 UTC (grid ends at
  2026-10-25 23:00 UTC). No imputation of PM2.5; missing stays NaN.
- Station-hours: 5 stations. Evaluation rows = (station, origin) with target, `pm25_latest` and
  `mean24` all present, so every candidate is scored on the identical set. Sample size reported.
- Weather: Open-Meteo **Previous Runs API** (`previous-runs-api.open-meteo.com/v1/forecast`),
  model `ecmwf_ifs025` throughout, variables with suffix `_previous_day1` (forecast issued about
  24 h before valid time): wind_speed_10m, wind_direction_10m, relative_humidity_2m,
  temperature_2m, precipitation, surface_pressure, cloud_cover. One request per station
  (5 requests), raw JSON cached under `data/raw/openmeteo/` (git-ignored), exact URLs recorded in
  `data/raw/openmeteo/requests.jsonl`. `_previous_day2` is fetched in the same request as a
  descriptive sensitivity only (a live run is not available at its init time, see caveat).
- Model choice: ECMWF IFS 0.25 deg is the highest-skill global model, has a documented archive of
  lead-specific runs since 2024, and one model keeps the features consistent. GFS is the fallback
  and is not used unless IFS fails to load.
- ERA5 reanalysis is NOT used as a feature (leakage vs live forecasts).
- Known caveat stated in advance: at 0.25 deg the 5 stations map to few grid cells, so station
  weather features are nearly collinear across stations. A forecast with a nominal 24 h lead is
  issued 24 h before valid time, but in live operation a run becomes available some hours after its
  init time, so the true live lead is longer (~30 h). The previous_day2 sensitivity brackets this.

## Candidates (all +24 h, direct, same rows)

1. Persistence: `pm25_latest` (latest reading in [origin-3 h, origin-1 h], 1 h reporting lag).
2. Smoother: `latest + beta * (mean24 - latest)`, beta in {0, 0.1, ..., 0.9} chosen per fold by min
   validation MAE on the last 10 days of that fold's training window.
3. LightGBM, PM2.5-only: features = KAN-65 set D (latest, calendar sin/cos, station_id,
   lags 24/48/72/168, mean/max/std over 3/6/24 h, trend3/trend6). No neighbour-station features.
4. LightGBM, PM2.5 + weather: set 3 plus, at the target hour, forecast wind_speed, wind_dir
   (sin, cos), relative_humidity, temperature, precipitation, surface_pressure, cloud_cover and the
   forecast change target-hour minus origin-hour for temperature, surface_pressure, wind_speed
   (11 weather features; all issued at or before the origin under the 24 h-lead construction).

LightGBM config fixed, no search: objective l1, num_leaves 7, learning_rate 0.05, 150 rounds,
min_child_samples 50, feature_fraction 0.9, deterministic, one thread; prediction = mean of 5 seeds
(20261029 + 0..4); pooled MAE spread across seeds reported. station_id categorical.

## Folds (chronological, expanding window, no overlapping test rows)

Test windows by origin time: G1 2026-09-01 to 09-22, G2 09-22 to 10-13, G3 10-13 to 10-26 (same
boundaries as KAN-65 G1-G3). Train rows: target index before the test window start (no label
overlap with test). Smoother validation = last 10 days of the training window. Pooled test rows of
the 3 folds are the evaluation set. First fold trains on only about 30 days; stated as a limit.

## Metrics, tests and decision rule

- Primary metric: MAE (ug/m3), paired per-row absolute error differences, pooled over folds.
- CI: percentile bootstrap, 10,000 draws, seed 20261030, resampling consecutive 7-day blocks of
  target date (about 8 blocks, so CIs are approximate; stated). 95% level.
- Primary comparison (one): weather LightGBM minus smoother, MAE difference.
- Secondary (descriptive, Bonferroni over 3 for CI level 1-0.05/3 also shown): weather vs
  persistence, weather vs PM2.5-only LightGBM (isolates weather), PM2.5-only LightGBM vs smoother.
- Spike onset: onset = y - latest >= 15 ug/m3; caught = forecast >= latest + 7.5 (KAN-65
  definition). Reported as caught share, false-alarm share among non-onsets, with paired 7-day-block
  CI for the difference between weather and PM2.5-only LightGBM. Judged only if onsets >= 30 on
  >= 8 distinct days, else "untestable".
- **Success (pre-declared):** (A) 95% 7-day-block CI of weather minus smoother MAE entirely below
  zero AND weather MAE gain vs persistence >= 15% (point estimate; CI reported), OR (B) onset caught
  share of weather model exceeds the PM2.5-only LightGBM by >= 10 percentage points with paired CI
  above zero and false-alarm share rising by <= 2 points.
- Otherwise: weather features do not clearly help in this sample. If CIs are wide (the 7-day
  CI half-width exceeds the effect needed), report "sample too small to decide" rather than "no
  effect".
- Recommendation mapping: success -> go (build ingestion); weather beats PM2.5-only LGBM with CI
  below zero but not the success rule -> pivot (longer history needed, e.g. a backfill); neither ->
  stop.
- Importance: LightGBM gain importance and mean |SHAP| (pred_contrib) of the weather features,
  fit on all pooled data (descriptive, not used for selection).

## Reproducibility

`kan75_fetch.py` (weather cache), `kan75_run.py` (all numbers into `out/kan75/results.json`),
run with the repo venv, `PYTHONPATH=src`. No hand-typed numbers.
