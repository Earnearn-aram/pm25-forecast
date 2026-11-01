# E07: classical time series (ARIMA, SARIMA, Holt-Winters) — KAN-85

Verdict: **DROP the family.** No variant beats the smoother; every paired difference versus the
smoother is positive (worse). Only the two ARIMA/ETS variants beat persistence with a CI below zero,
which the smoother already does with a lower MAE. Dev split only; the holdout was never touched.

## 1. Question and decision rule (preregistered before any run)

Preregs: `docs/experiments/prereg/e07-*.yaml` (5 files, committed in `361b1a1` before the first
logged run; every run is clean, `git_dirty=false`). Question: does a per-station classical model
that uses only the station's own hourly history lower +24 h MAE versus the smoother (KAN-76)?
Rule: keep the family only if a variant has `diff_vs_smoother_24h <= -0.15` and a 72 h block CI
entirely below zero against both persistence and the smoother after the 5-variant correction, and
is not fragile. Primary CI block length: **72 h** (24 h and 168 h as sensitivities; B 10,000, seed
20261101). Nothing else is exploratory: the 5 variants are all that was run.

## 2. Setup

- Snapshot `e02-v1` (hash `a00cfc9e...`), dev, 5 weekly folds, 5 stations, 4,200 station-hours at
  +24 h on 840 hourly origins. Air4thai era, targets before 2026-10-26. `available_at = ts + 1 h`
  (protocol section 2 assumption). No CAMS: E10 is not ready and the default snapshot has no vintages, so SARIMAX with CAMS
  is skipped.
- Code: `src/pm25_forecast/experiments/timeseries.py` (factories `arima`, `sarima`, `ets`),
  tests `tests/test_e07_timeseries.py`. Results table: `python -m pm25_forecast.experiments.e07_summary`.
- **Serving lag.** The series for origin `t` ends at the grid hour `t - 1 h` and uses only rows
  with `ts <= t - 1 h` and `available_at <= t`; the forecast for `t + h` is step `h + 1` from that
  point. Tested (a later reading is rejected; corrupting rows at or after `t` changes nothing).
- **Gaps.** Hourly grid, missing hours are NaN and the target is never filled. ARIMA/SARIMA are
  state-space models, so the Kalman filter skips a missing hour and carries the state (no value is
  invented). Holt-Winters cannot take NaN, so it uses the newest gap-free run (at most 168 h) and returns no
  forecast when that run is under 72 h. As for the baselines, no forecast when the newest reading
  is older than 3 h. The frozen snapshot has no missing hours in the cohort (manifest), so gap
  handling is exercised by the unit tests only, not by the dev numbers.
- **Refit and update.** Parameters and (for ARIMA/SARIMA) the order are re-estimated every week on
  the training part of the fold only (AIC; the runner refits 24 h before the fold). Between refits
  the filter state is extended by one hour at a time with frozen parameters (`results.extend`,
  about 3 ms); Holt-Winters re-runs its recursion over the last 168 h with frozen smoothing parameters.
- Variants (5): **arima-d0** (constant, p 0-3, q 0-2, expanding window), **arima-d1** (p 0-3, q 0-2),
  **sarima** (24 h season; p=1, q 0-1, seasonal P/Q in {(1,0),(0,1),(1,1)}, newest 42 days),
  **ets-damped** (additive damped trend + additive 24 h season), **ets-notrend** (level + season).
  The SARIMA grid is deliberately small to fit the runtime budget; a wider grid was not tried.

## 3. Results (+24 h, dev; unadjusted 95% CIs, 72 h blocks)

Baselines on the same 4,200 pairs: persistence MAE 4.311, smoother MAE 3.867.

| variant | run | mae_24h | mae_6h | mae_12h | diff vs persistence [CI] | diff vs smoother [CI] |
|---|---|---|---|---|---|---|
| arima-d1 | ba06af54 | 3.965 | 3.420 | 3.670 | -0.347 [-0.654, -0.145] | +0.098 [-0.069, +0.225] |
| ets-notrend | 1e4a04a1 | 4.159 | 3.884 | 4.134 | -0.152 [-0.225, -0.080] | +0.292 [+0.153, +0.486] |
| ets-damped | be5b6721 | 4.172 | 3.886 | 4.140 | -0.139 [-0.218, -0.058] | +0.305 [+0.172, +0.495] |
| arima-d0 | ce1c4b36 | 4.176 | 3.597 | 3.917 | -0.135 [-0.646, +0.281] | +0.309 [-0.173, +0.770] |
| sarima | e6fec9f7 | 4.411 | 3.738 | 4.183 | +0.100 [-0.655, +0.618] | +0.544 [-0.193, +1.121] |

Best variant: **arima-d1**, MAE 3.965, skill vs persistence +8.0%. Versus the smoother it is
**worse** by 0.098 ug/m3; the CI includes zero (sensitivities 24 h [-0.144, +0.292], 168 h
[-0.090, +0.216], not fragile), and the sign is positive in 4 of 5 folds. Versus persistence its CI is
entirely below zero, but the smoother does the same with a lower MAE, so this does not count. Both
ETS variants are significantly worse than the smoother (CI entirely above zero). At +6 h arima-d1
(3.420) is close to the smoother's level but still not better (diff +0.050, CI [-0.091, +0.177]);
no horizon is gating.
Nothing beats the smoother, with or without correcting for 5 variants (no CI is below zero, so a
Bonferroni level cannot change that); no corrected run was needed. Spike metrics are undefined
(15 observed spikes, under 50). No negative forecasts, no missing forecasts (candidate_none 0).

## 4. Runtime

Weekly refit time per fold (5 folds, seconds; from `PM25_TS_TIMING_LOG` files in `~/pm25-experiments/e07-timing`):
arima-d0 9.3, 9.9, 8.5, 9.6, 12.5 (total 49.8); arima-d1 4.7, 3.7, 3.5, 4.6, 4.2 (20.8);
sarima 40.8, 44.3, 53.7, 53.6, 53.0 (245.4); ets-damped 0.7 per fold (3.5); ets-notrend 0.2 per
fold (0.9). Whole dev run including baselines and 10,000-resample bootstraps: arima-d0 69 s,
arima-d1 31 s, sarima 258 s, ets-damped 21 s, ets-notrend 16 s. SARIMA is the only one beyond a
few minutes, and it is the worst model; the hourly update is cheap for all of them.
Orders chosen by AIC are in the timing files: ARIMA d0 mostly (2,0,2); ARIMA d1 mostly (1,1,2);
SARIMA nearly always (1,0,1)(1,0,1)24.

## 5. Interpretation and limits

- Possible reasons (not tested): the series is low level (mean about 13 ug/m3, sd about 5 at station 02t) and the 24 h ahead
  signal is mostly mean reversion, which the smoother captures with one fixed parameter. Seasonal
  models gain nothing at +24 h and SARIMA is worse than non-seasonal ARIMA d1, probably through
  over-fitting the seasonal terms on 42 days (an explanation, not tested).
- Limits: one 12-week window, five stations, one season; the SARIMA order grid and 42-day window
  are small and fixed in advance; no log transform or exogenous inputs (CAMS, weather) were
  tried; AIC was computed on the (capped) training sample only. These are descriptive dev
  results, not evidence about the holdout.

## 6. Recommendation

Drop classical time series as a stand-alone family (confidence: high for "does not beat the
smoother" on this data, medium for generality). The only reason to revisit is as a component:
arima-d1 as a feature or stacking input, or SARIMAX once CAMS vintages exist (E10).
