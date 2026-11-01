# E06: linear models (OLS, ridge, lasso), KAN-84

Verdict: **no linear variant beats the smoother.** The best one, pooled ridge predicting the change
from the latest reading on a raw target, is clearly better than persistence and 0.110 ug/m3 better
than the smoother in point estimate (all numbers below are from the re-run after the purge fix, section 8), but its CI against the smoother spans zero (also at the
multiplicity-adjusted level). Dev split only; the holdout was never touched.

## 1. Question, hypothesis, rule (fixed in git before the runs)

Preregs: `docs/experiments/prereg/e06-*.yaml` (24 grid files committed in `7970c52` before any
logged run; the sensitivity file in `4307fab` before its run). Question: does any linear model on
the shared `fs_v2_core` features lower +24 h dev MAE versus the smoother, and which variant is best?
Rule: the best variant is the valid one with the lowest dev `mae_24h`. It counts as better than the
smoother only if its **72 h block-bootstrap CI at the Bonferroni level `1 - 0.05/25` (K = 25) is
entirely below zero**. The unadjusted 95 % CI is descriptive. 72 h is the primary CI block (24 h and
168 h are sensitivities, protocol section 6).

Disclosure: before the preregs, one `--no-log` timing run of `ridge-pool-raw-lvl` (same code) was
executed to measure runtime; its 3.768 MAE was seen. No setting was changed afterwards. During the
re-run, two `--no-log` runs of `ridge-pool-raw-chg` diagnosed the serving-lag probe failure
(section 8); no hypothesis or decision rule changed.

## 2. Setup

- Snapshot `e02-v1` (hash a00cfc9e...), dev split, 5 weekly folds, weekly refit, 5 stations,
  4,200 station-hours at +24 h on 840 origins (all 24 variants cover 100 % of baseline-eligible
  pairs); `available_at = ts + 1 h` assumption of protocol section 2.
- Features: `fs_v2_core` from `features/library_v2.py` (lags 1-48 h, rolling means 3-72 h, 1 h and
  24 h differences, calendar), 25 columns (plus 5 station one-hots when pooled). No private feature
  code. A row exists only when a reading is present in `[t-3h, t-1h]` (same rule as persistence);
  other missing inputs are filled with the training mean, then standardised.
- Models: `src/pm25_forecast/experiments/linear_models.py`. Regularisation is chosen by an inner
  expanding-window CV **inside the training part of each weekly fold**: 5 time blocks by origin,
  4 splits, a training row is used only if its label was available (target time + 1 h) before the
  validation block starts, imputation and scaling are
  refitted per split, criterion MAE in ug/m3 after back-transform. Grids (frozen): ridge lambda
  1e-4..1 (9 half-decades; penalty = n x lambda on standardised features), lasso alpha =
  ratio x alpha_max with ratio 1e-4..1. The intercept is not penalised; station one-hots are.
- Grid of **24 variants**: {ols, ridge, lasso} x {pooled with station one-hot, one model per
  station} x {raw, log1p target} x {level, change from latest}. Log1p is back-transformed with
  `expm1` (no bias correction); "change" is `y - latest` (raw) or `log1p(y) - log1p(latest)`; latest
  is the persistence value. Negative predictions are clipped by the runner (0 were clipped).
- **Multiple comparisons:** 24 grid variants + 1 pre-declared sensitivity (best variant on
  `fs_v2_full` without its cross-station group, which a per-station predictor cannot see) = **K = 25**
  candidates compared with the smoother. Dev CIs in the table are unadjusted; the decision uses the
  Bonferroni interval of the best variant. Total tracked runs in the re-run (the ones used here): 24 + 1 sensitivity + 1 Bonferroni
  re-computation of the best variant (same predictions, `--finalists 25`, experiment
  `kan78/kan84-bonf25`, run d4917312) = 26. OLS "level" and "change" on a raw target are the same linear model
  (the latest reading is a linear function of the lags), so they have identical scores; they are
  still counted.

## 3. Results (+24 h, 4,200 pairs; diff = candidate - baseline, negative is better; 72 h CI)

Baselines (E04/E05, same rows): persistence MAE 4.311; smoother MAE 3.867.

| prereg | run | MAE24 | skill vs pers. | vs persistence | vs smoother |
|---|---|---|---|---|---|
| e06-ridge-pool-raw-chg (best) | 92151830 | 3.757 | +0.129 | -0.554 [-0.833, -0.311] | -0.110 [-0.311, +0.133] |
| e06-ridge-pool-raw-lvl | c4d10308 | 3.768 | +0.126 | -0.544 [-0.852, -0.264] | -0.099 [-0.328, +0.183] |
| e06-lasso-pool-raw-chg | f192152a | 3.787 | +0.122 | -0.524 [-0.803, -0.271] | -0.080 [-0.295, +0.183] |
| e06-lasso-stn-raw-chg | b7277fac | 3.797 | +0.119 | -0.515 [-0.788, -0.304] | -0.070 [-0.313, +0.180] |
| e06-ridge-pool-raw-chg-full (sensitivity) | 2699abb8 | 3.797 | +0.119 | -0.514 [-0.758, -0.288] | -0.070 [-0.241, +0.147] |
| e06-lasso-pool-raw-lvl | 2e3445a3 | 3.802 | +0.118 | -0.510 [-0.881, -0.202] | -0.065 [-0.357, +0.246] |
| e06-ridge-stn-raw-chg | a4dcdb3f | 3.814 | +0.115 | -0.497 [-0.785, -0.254] | -0.053 [-0.297, +0.225] |
| e06-ols-pool-raw-chg = -lvl | ba388eab / 70fac138 | 3.829 | +0.112 | -0.483 [-0.871, -0.163] | -0.039 [-0.364, +0.300] |
| e06-ridge-pool-log-lvl | 3d3c3c4f | 3.829 | +0.112 | -0.482 [-0.812, -0.208] | -0.038 [-0.296, +0.265] |
| e06-ols-stn-raw-chg = -lvl | 997f2ef0 / 8cc75e40 | 3.849 | +0.107 | -0.462 [-0.849, -0.154] | -0.018 [-0.347, +0.309] |
| e06-ridge-stn-raw-lvl | 04784a33 | 3.862 | +0.104 | -0.449 [-0.839, -0.121] | -0.005 [-0.337, +0.351] |
| remaining 12 variants (all log1p except lasso-stn-raw-lvl) | | 3.913 - 4.251 | +0.092 - +0.014 | | +0.046 - +0.384, all CIs span 0 |

All 24 run ids and numbers: `python -m pm25_forecast.experiments.kan84_analyze table`.

Best variant, in detail:
- MAE 3.757 (RMSE 5.193 +24 h; 3.339 at +6 h, 3.530 at +12 h), skill +12.9 % over persistence.
- Versus persistence: -0.554 [-0.833, -0.311] (95 %), below zero in 5 of 5 folds, and every station
  improves (by 8.4 % to 19.5 %). Station gate (no station more than
  5 % worse than persistence): **passes**; per station MAE (model / persistence): 02t 3.130 / 3.487
  (-10.2 %), 59t 2.959 / 3.232 (-8.4 %), 61t 3.095 / 3.418 (-9.4 %), bkp100t 4.694 / 5.324 (-11.8 %),
  bkp101t 4.906 / 6.097 (-19.5 %). Three log1p-change per-station variants fail the gate on bkp100t
  (ridge +6.2 %, lasso +13.1 %, ols +8.3 %); none is a contender.
- Versus the smoother: -0.110 [-0.311, +0.133] (95 %); sensitivities 24 h [-0.407, +0.178], 168 h
  [-0.345, +0.048]; DM p = 0.32 (m = 167); below zero in 3 of 5 folds. **Bonferroni (K = 25, 99.8 %)
  CI: [-0.410, +0.274]** (run d4917312; the first-round run fa8b1290 gave the same interval). It does not beat the smoother. The minimum effect of the
  promotion gate (-0.15) is not reached either.
- Spike metrics are undefined (15 observed spikes, below 50; recall and precision not logged).

## 4. Reading the variants

- Raw target beats log1p in every pair (the 12 worst variants are log1p, plus lasso-stn-raw-lvl).
  Predicting the median of a skewed target after `expm1` does not help MAE here, and "change in
  log space" is the worst choice.
- Level versus change matters little for ridge (3.768 vs 3.757) and nothing for OLS (identical);
  it helps lasso (3.802 to 3.787) and per-station ridge (3.862 to 3.814) plausibly because shrinkage then
  pulls toward persistence rather than toward the training mean.
- Pooled with station one-hot is as good as or better than one model per station (ridge 3.757 vs
  3.814) at 5 stations and about 4 weeks of data at the first refit.
- Ridge >= lasso >= OLS in the best configurations; the differences (0.03 to 0.07) are far inside the
  CI noise, so the ranking among the top variants is not established. The selection among 24 is
  optimistic: the best variant's own gain over the smoother would shrink on new data.
- `fs_v2_full` without cross-station features is not better (3.797 vs 3.757).

## 5. Coefficients of the best variant (pooled ridge, change from latest, +24 h)

Last refit (2026-09-27, 6,750 rows; lambda 0.316 chosen in every refit of the re-run, so inside the
grid and unchanged by the purge fix). Values
are per 1 SD of the feature, in ug/m3 added to the latest reading:
`pm25_lag_1h` -1.08, `pm25_diff_1h` -0.85, `pm25_diff_24h` -0.55, `pm25_mean_3h` -0.55,
`pm25_lag_2h` -0.50, `dow_cos` +0.38, `station_bkp100t` -0.37, `station_bkp101t` +0.36,
`is_weekend` +0.32, `hour_cos` -0.31; the rest are below 0.21 in magnitude (`python -m
pm25_forecast.experiments.kan84_analyze coef e06-ridge-pool-raw-chg`).
The lag, mean and difference features are strongly collinear, so individual signs are not
interpretable. Summed in raw units, a uniform +1 ug/m3 shift of the whole history moves the +24 h
forecast by about +0.65 ug/m3 (0.59 to 0.80 across refits): the model is partial mean reversion
to the recent level, similar in spirit to the smoother, plus small calendar and station offsets
(weekend, day-of-week, hour of day). Lasso (ratio 0.03 to 0.1 in the re-run) keeps 10 to 16 of 25 features, mostly
`pm25_lag_1h` and the differences. These are predictive associations, not causes.

## 6. Limits

- About 9 weeks, one monsoon-to-dry-season transition, 5 stations, 840 origins: 3 to 5 independent
  weeks. Dev only, no holdout, K-adjusted CIs are the only inference that counts.
- Serving lag: the runner's automated probe (KAN-96, gate 10.4) re-predicts 3,275 sampled
  (origin, station, horizon) cases with all forbidden rows added and shifted by +1000 and -999 on
  fresh predictors. For the best variant (and all 26 re-run runs): **pass, 0 mismatches, 0
  errors**. The unit tests additionally show that features at origin `t` ignore readings at
  `ts >= t`, that `latest` equals the shared `build_features` value, and that inner CV never
  trains on a label not yet available (target + 1 h). The runner's `Snapshot` also rejects data
  after the origin.
- The cross-station features are untested because the runner only gives a predictor its own
  station's readings.
- Registry loading (`load_serving_model`) was not done; this is a dev experiment.

## 7. Reproduce

    for f in docs/experiments/prereg/e06-*.yaml; do
      python -m pm25_forecast.experiments.evaluate --prereg $f \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 \
        --model pm25_forecast.experiments.linear_models:fit
    done
    python -m pm25_forecast.experiments.evaluate --prereg docs/experiments/prereg/e06-ridge-pool-raw-chg.yaml \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 --model pm25_forecast.experiments.linear_models:fit \
        --finalists 25 --experiment kan78/kan84-bonf25 --run-name ridge-pool-raw-chg-bonf25
    python -m pm25_forecast.experiments.kan84_analyze table

Generate the preregs with `python -m pm25_forecast.experiments.kan84_prereg` (and `--sensitivity
e06-ridge-pool-raw-chg`). Seeds: none needed (closed-form ridge and OLS, deterministic cyclic lasso);
bootstrap seed 20261101. Diagnostics per refit: `~/pm25-experiments/cache/kan84/<variant>.jsonl`.
The leaderboard file `docs/experiments/leaderboard.md` was not regenerated in this PR (it merges
every agent's runs and would conflict); regenerate after the E0x PRs land.

## 8. Deviations and superseded runs

The first round of runs (24 variants, 1 sensitivity, Bonferroni run fa8b1290; their MLflow runs
remain, e.g. best variant a64d43ca) is **superseded**. Two code defects were found in review and
by the KAN-96 probe, so the whole grid was re-run on the same preregs (no hypothesis or decision
rule changed; lambda values quoted here come from the re-run):

1. The inner-CV purge was 1 h short: a label with target `t = origin + h` is available at
   `t + 1 h`, so training rows need `origin + (h + 1) h <= block start` (same fix as KAN-89). Effect:
   small. Best variant and its numbers are identical to the first round (lambda 0.316 in every
   refit), and the other variants moved by at most 0.017 MAE (for example lasso-pool-raw-lvl
   3.819 to 3.802, ridge-stn-raw-lvl 3.859 to 3.862); the ranking at the top is unchanged.
2. `predict` took the last 80 snapshot rows by position, so rows after the origin (which the
   probe adds) displaced the history and returned no forecast: the KAN-96 probe failed (3,270
   of 3,275 checks) on the first attempt of the re-run. It now filters by time. Live scores are
   unaffected (the runner's snapshot never held future rows), the probe passes.

Commits: purge fix `898861f`, time filter `a75c5db`, runs on `7b873a8` (clean tree, `git_dirty`
false). The leaderboard and `kan84_analyze table` use the newest valid run per experiment, so the
re-run is what they show.
