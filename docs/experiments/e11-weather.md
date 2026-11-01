# E11: weather-forecast features (KAN-88)

Verdict: **PIVOT**. Weather forecasts do not lift any model past the smoother or past 15 % over
persistence, so the KAN-75 success rule is not met. They do help LightGBM (about 0.31 ug/m3 lower
+24 h MAE; CI below 0 overall, negative in all 5 folds) but not the linear model, and the helped LightGBM is still no
better than the smoother. No live-ingestion ticket is written (it is for "go" only).

## 1. Question, hypothesis, decision rule (fixed before the runs)

Preregs: `docs/experiments/prereg/e11-*.yaml` (6 files, committed in `c87259d` before any logged run).
Question: do lead-matched NWP forecast features lower +24 h MAE of LightGBM and the best linear
model on identical rows, and does that meet the KAN-75 success rule?
Rule (A) 95 % CI of weather minus smoother MAE entirely below 0 and MAE gain versus persistence at
least 15 %, or (B) spike-onset catch share at least 10 points above the no-weather twin with paired
CI above 0 and false alarms up at most 2 points (judged with at least 30 onsets on at least 8 days).
Go = (A or B) for a family and the weather increment over its no-weather twin has a 97.5 % CI
(Bonferroni over the 2 families) entirely below 0. Pivot = not go but a negative increment point
estimate in a family. Stop = otherwise. Dev split only; the holdout was never touched.

## 2. Setup

- Snapshot `e02-v1` (hash a00cfc9e...), dev split, 5 weekly folds, weekly refit, 5 stations, 4,200
  station-hours at +24 h on 840 hourly origins; air4thai era, targets before 2026-10-26.
  `available_at = ts + 1 h` assumption of protocol section 2 applies.
- Base features: `fs_v2_core` (lags, rolling means, diffs, calendar). Weather: add-on set `wx_v1`
  in `src/pm25_forecast/features/weather.py` (12 features at the target hour: wind speed, wind
  direction as sin/cos, humidity, temperature, rain, surface pressure, cloud cover, shortwave
  radiation, plus 24 h rain sum, wind-speed mean and shortwave sum).
- Weather source: Open-Meteo Previous Runs API, ECMWF IFS 0.25 deg only, `*_previous_day1` at the
  station coordinates; cache in `~/pm25-experiments/cache/weather` (outside git, 5 files, request
  URLs in `requests.jsonl`). No ERA5 or analysis values. Sensitivity: `*_previous_day2`.
- Boundary-layer height was requested but the API returns null for every hour (also for GFS and
  the other ECMWF model), so it is **not** a feature. Shortwave radiation stands in as the mixing
  driver. Wind direction is encoded as sin/cos.
- Grid: the 5 stations fall on 3 distinct 0.25 deg cells (02t, 59t, 61t share one; bkp100t and
  bkp101t share one), so station weather is nearly collinear.
- Models: ridge (alpha 30, standardised, mean-filled) and LightGBM (l1, 7 leaves, 150 rounds, lr
  0.05, min child 50, mean of 5 seeds), one model per station and horizon, hyper-parameters frozen
  in the prereg, no tuning. "Best linear model": KAN-83/84 are not on `main`, so this is ridge on
  `fs_v2_core`, chosen in advance, not the best of a search. Swap it in if KAN-84 differs.
- Identical rows: every variant, with and without weather, returns no forecast when the weather
  inputs of the target hour are missing; all variants scored on the same 4,200 pairs.
- Serving-lag: a weather value for hour u with lead L days is used at origin t only if
  u - 24 h x L <= t (tested). Caveat: a real run is published hours after init, so `previous_day1`
  at the +24 h target hour (nominal issue time = origin) is a slightly optimistic stand-in for what
  is live at t. `previous_day2` (issued 24 h before the origin) is strictly available.

## 3. Results

Identical rows: 4200 station-hours at +24 h over 840 hourly origins, stations 02t, 59t, 61t, bkp100t, bkp101t. MAE in ug/m3; diff = MAE(row) - MAE(reference), negative is better; brackets are the 72 h block-bootstrap CI.

| model | MAE | vs persistence (95%) | vs smoother (95%) | skill vs persistence |
|---|---|---|---|---|
| persistence | 4.311 |  |  | +0.000 |
| smoother | 3.867 |  |  | +0.103 |
| ridge_base | 3.838 | -0.473 [-0.855, -0.161] | -0.029 [-0.352, +0.301] | +0.110 |
| ridge_wx | 3.885 | -0.426 [-0.883, +0.004] | +0.018 [-0.415, +0.452] | +0.099 |
| lgbm_base | 4.282 | -0.029 [-0.623, +0.339] | +0.415 [-0.146, +0.837] | +0.007 |
| lgbm_wx | 3.970 | -0.342 [-0.895, +0.010] | +0.103 [-0.437, +0.522] | +0.079 |
| ridge_wx_d2 | 3.874 | -0.438 [-0.743, -0.100] | +0.007 [-0.314, +0.406] | +0.102 |
| lgbm_wx_d2 | 3.958 | -0.354 [-0.826, -0.039] | +0.091 [-0.369, +0.469] | +0.082 |

Weather increment, same family and same rows (wx minus base):

| family | variant | diff, 95% CI | diff, 97.5% CI (Bonferroni, 2 families) | by fold | fragile |
|---|---|---|---|---|---|
| ridge | day1 (primary) | +0.047 [-0.224, +0.341] | +0.047 [-0.258, +0.380] | +0.01, +0.05, +0.07, +0.33, -0.23 | False |
| ridge | day2 (strict availability) | +0.036 [-0.282, +0.446] |  | -0.47, +0.39, +0.73, -0.22, -0.26 | False |
| lgbm | day1 (primary) | -0.312 [-0.458, -0.163] | -0.312 [-0.478, -0.141] | -0.57, -0.24, -0.19, -0.52, -0.04 | False |
| lgbm | day2 (strict availability) | -0.324 [-0.532, -0.053] |  | -0.77, -0.08, -0.15, -0.49, -0.14 | False |

Spike onsets (y - latest >= 15): 64 on 13 days, testable=True.
- ridge_base: caught 0.078125, false-alarm share 0.0097
- ridge_wx: caught 0.125, false-alarm share 0.0176
- lgbm_base: caught 0.0625, false-alarm share 0.0104
- lgbm_wx: caught 0.078125, false-alarm share 0.0126

Pre-declared KAN-75 rule and mapping:

- ridge_wx: A(CI vs smoother < 0) = False, gain vs persistence +0.099 [-0.001, +0.205] (needs >= 0.15) -> A = False; B = False; increment CI(97.5%) < 0 = False; go = False
- lgbm_wx: A(CI vs smoother < 0) = False, gain vs persistence +0.079 [-0.002, +0.208] (needs >= 0.15) -> A = False; B = False; increment CI(97.5%) < 0 = True; go = False

Verdict by the pre-declared mapping: **pivot**

MLflow (experiment `kan78/e11-<variant>`, tags ticket KAN-88, agent claude-data-scientist, git
`c87259d`, clean): ridge_base a347ab56, ridge_wx 1cebc1f1, lgbm_base edc52985, lgbm_wx e41b2f96,
ridge_wx_d2 6ae316d1, lgbm_wx_d2 62adf148 (id prefixes).

## 4. Reading

- Ridge: weather adds nothing (+0.05 ug/m3, CI spans 0); the no-weather ridge is already as good
  as the smoother (-0.03, CI spans 0).
- LightGBM: weather lowers MAE by 0.31 (CI [-0.46, -0.16], also at 97.5 %), negative in all 5 folds
  and with `previous_day2`. But the no-weather LightGBM is the worst model here (worse than the
  smoother by 0.42), and with weather it is still above the smoother (+0.10, CI spans 0). Weather
  repairs part of a weak model; it does not create a model that beats the baselines.
- Not a causal claim. With three grid cells and only about nine weeks (one monsoon-to-dry-season
  transition), slow-moving weather features can act as a date or regime proxy, which a per-station
  tree model could exploit without any real weather effect. A longer history would separate those.
- Spike onsets: 64 on 13 days; catch rates are 6-13 % for every model, so spike detection is not
  improved in any usable way (rule B not met; ridge +4.7 points is far from the 10-point bar).
- Dev CIs are unadjusted for the 6 runs; this is exploratory and no holdout evidence.

## 5. Decision

Pivot. Do not build live ingestion of previous-run forecasts now. The condition to revisit is a
longer history (the Previous Runs archive reaches back to 2024; a backfill of the PM2.5 side would
be needed to use it) so that the LightGBM gain can be tested against a season-aware baseline. If
KAN-89 (LightGBM on `fs_v2_*`) reaches a strong no-weather LightGBM, re-test the weather increment
on top of it with `kan88_analyze` before any ingestion work.

## 6. Reproduce

    python -m pm25_forecast.experiments.kan88_fetch_weather          # once, public API
    for v in ridge_base ridge_wx lgbm_base lgbm_wx ridge_wx_d2 lgbm_wx_d2; do
      python -m pm25_forecast.experiments.evaluate --prereg docs/experiments/prereg/e11-${v//_/-}.yaml \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 --model pm25_forecast.experiments.kan88_models:$v
    done
    python -m pm25_forecast.experiments.kan88_analyze --snapshot ~/pm25-experiments/snapshots/e02-v1

Seeds: LightGBM 20261101 + 0..4, bootstrap 20261101. Process note: one unlogged `--no-log` smoke run
of `ridge_wx` (MAE 3.885, the same as the logged run) was made to check the plumbing before the
preregs were committed; the models and rules were not changed afterwards.
