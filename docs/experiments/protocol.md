# Experiment protocol (KAN-79, story KAN-78)

Single source of truth for every experiment under KAN-78. Fixed by Boss on 2026-11-01. Do not
deviate silently: a deviation makes the run invalid (section 12) unless this file is amended in a
reviewed PR first. Words MUST / MUST NOT are binding.

**Goal.** Find a +24 h PM2.5 model that beats persistence **and** the smoother on a locked
holdout, with a paired 95% CI entirely below zero, servable through the existing registry and
shadow path.

## 1. Targets, horizons, baselines

- Target: hourly PM2.5 (µg/m³) of a station at `t + h`. Primary horizon **+24 h**; secondary +6 h
  and +12 h (reported, never gating).
- **Persistence**: newest observation with `ts` in `[t−3h, t−1h]` (the 3 h age limit of
  `models/baselines.py`, applied after the serving lag); none, no forecast.
- **Smoother**: `latest + 0.5 · (mean24 − latest)`, `latest` as above, `mean24` the mean of the
  observations with `ts` in `[t−24h, t−1h]` (at least 12 of 24). The same formula is used at every
  horizon (KAN-76, PR #59). If the merged KAN-76 code differs, the code wins and this line is
  amended.
- Candidate, persistence and smoother are always scored on the **common intersection** of
  (station, origin, horizon). No imputation. A candidate that covers fewer than 95% of the
  baseline-eligible pairs of a split is invalid for that split.

## 2. Serving lag (leakage rule)

For origin `t`, every feature input MUST satisfy `ts <= t − 1h` **and** `available_at <= t`
(forecast vintages: `issue_time` and `available_at <= t`). The observation of hour `t` is never
used. The same rule applies to training rows: a training pair (origin `o`, target `o+h`) is usable
at a refit time `T` only if `o + h <= T` and the label `available_at <= T`. Features come from
`src/pm25_forecast/features/` (shared with serving); no private feature code. Rows without a
valid `available_at` are excluded, never filled with `ts`.

## 3. Data

- Frozen snapshot from E02 (KAN-80), identified by `snapshot_sha256`. Never re-export mid-story.
- **air4thai era only**: `ts >= 2026-07-31T17:00:00Z` (`S0`). Earlier openaq rows MUST NOT enter
  features, labels or training (source break: openaq is ~8.6× smoother hour to hour; KAN-65).
- **Targets after 2026-10-26T00:00:00Z are excluded** until the KAN-74 repair; rows after that may
  not be used as labels. (Feature inputs before the cut are fine.)
- Station cohort: stations in the snapshot with at least 80% hourly coverage in the dev period,
  fixed once in the snapshot manifest and used by all experiments.
- All times UTC, timezone-aware.

## 4. Walk-forward design

- Hourly origins; **expanding window** (all usable pairs since `S0`); **weekly refit**; the
  model fitted at refit time `T` forecasts every origin in `[T, T+7d)`.
- Training uses only pairs usable at `T` (section 2). Hyper-parameters are frozen in the
  pre-registration; refits re-estimate parameters only. No tuning on holdout folds.
- Initial training window: at least **28 days** after `S0`.

### 4.1 Deriving fold dates from the snapshot

Let `L` = last target time in the snapshot that passes the exclusions in section 3 (at most
2026-10-25T23:00Z while the cut applies). `E` = `L + 1h` floored to UTC midnight: the exclusive
end of the last full UTC day with targets. Then:

1. Holdout targets: `[H0, E)` with `H0 = E − 21 d` (UTC midnight). Three weekly folds
   `[H0, H0+7d)`, `[H0+7d, H0+14d)`, `[H0+14d, E)`, each refit at its start.
2. Dev folds: weekly steps backwards from `H0`: `[H0−7d·k, H0−7d·(k−1))`. Keep a fold only if its
   start is `>= S0 + 28 d`; a shorter leftover at the start is dropped (it is training data).
3. Origins of a fold are those whose **targets** lie in the fold, per horizon.
4. Dev targets never include holdout targets. Dev **training** rows must have their target `< H0`.
   Holdout folds train on everything matured before their refit time, including earlier holdout
   weeks (that is honest walk-forward). The holdout labels MUST NOT be read by any dev run: dev
   code takes `split=dev` data only (the snapshot loader enforces `target < H0`).

Fold dates are recomputed from the snapshot by one shared function and logged as run params
(`H0`, `E`, `fold_starts`); they are never typed by hand.

### 4.2 Locked holdout and short snapshots

- Holdout = **last 21 days** of targets. The data-scientist or review agent unlocks it only for
  preregistered finalists (section 9).
- **Minimum dev length**: `D = H0 − S0 >= 49 d` (28 d initial training + at least 3 weekly dev
  folds). Hence total usable span `E − S0 >= 70 d`.
- **If `D < 49 d`** (snapshot too short): the holdout is **not** carved out, no run may carry
  `split=holdout`, no promotion decision is possible. Experiments run dev-only (all folds after
  `S0 + 28 d`), results are exploratory, and the story waits for more data (re-freeze with a new
  `snapshot_sha256` and restart the preregistrations). The holdout is never shortened below 21 d
  and the dev period never shrunk to make room.
- At the 2026-10-26 cut the snapshot gives `E − S0 ≈ 87 d`, `D ≈ 66 d`: the holdout exists.

## 5. Metrics

All computed on the common intersection (section 1), per horizon, from code in the repo.

- `mae_24h`, `rmse_24h` (`_6h`, `_12h` analogues): over all station-hour pairs.
- `skill_vs_persistence_24h = 1 − MAE_model / MAE_persistence`.
- `diff_vs_smoother_24h = MAE_model − MAE_smoother` (and `diff_vs_persistence_24h`). **Negative is
  better.** Computed as the mean of paired absolute-error differences.
- `n_pairs_24h`: number of station-hour pairs in the intersection.
- Thai PM2.5 bands (µg/m³, upper edges): 0–15 very good, 15–25 good, 25–37.5 moderate,
  37.5–75 starting to affect health, >75 affecting health. A value equal to an edge falls in the
  lower band (same as `backtest.py`; edges in `retrospective.CATEGORY_EDGES`). Category accuracy
  is an extra diagnostic, not a gating metric.
- **Spike** = value `> 37.5` (strictly above, consistent with the lower-band rule). A pair is a
  true positive if observed and predicted are both spikes; false negative if only observed;
  false positive if only predicted.
  `spike_recall_24h = TP/(TP+FN)`, `spike_precision_24h = TP/(TP+FP)`. If the denominator is
  zero, or the split has fewer than **50 observed spike pairs**, log no value (null) and report the
  counts; never log 0 or 1 for an undefined case. Spike metrics are diagnostics, not gates.
  (KAN-65: spike detection was very weak, ≤ 10% recall; report it honestly.)
- Each metric is logged with the `_6h`/`_12h` variants for every horizon in the preregistration.

## 6. Statistics

Unit of analysis for inference: the **hourly series** `d_t` = mean over stations of the paired
absolute-error difference (candidate − baseline) at origin hour `t`. This keeps cross-station
dependence inside one observation. Pairs from consecutive origins overlap in time and
overlapping horizon errors are autocorrelated: pairs MUST NOT be treated as independent.

- **Paired MAE difference with 7-day moving-block bootstrap CI.** Blocks are 168 consecutive
  hourly origins on the calendar grid, start uniformly at any grid hour that fits; each block
  carries (sum of pair differences, pair count); draw `ceil(n_hours/168)` blocks with replacement,
  statistic = total sum / total count (handles gaps). **B = 10,000 resamples, seed 20261101**,
  `numpy.random.default_rng(seed)`, percentile interval. `diff_vs_*_ci_low/_ci_high` are the
  bounds at the level in section 8 (95% for one finalist), logged as `ci_level` param.
  Mandatory sensitivity (report, not gating): block lengths 24 h and 72 h. If the 24 h
  interval crosses 0 when the 7-day one does not, the report MUST mark the result *fragile*
  (KAN-65: borderline CIs flipped with block size).
- With a 21 d holdout there are only ~3 independent 7-day blocks: say so in every holdout report.
- **Diebold–Mariano** on the same `d_t` series (absolute-error loss, so it tests MAE equality),
  versus the smoother, two-sided: `dm_pvalue_vs_smoother_24h`. Long-run variance by
  Newey–West (Bartlett kernel) with bandwidth **`m = 2h − 1 = 47` for h = 24** (11 for h = 6, 23 for
  h = 12): `h − 1` is the minimum MA order implied by h-step overlap; doubling it covers the daily
  cycle in the errors. Harvey–Leybourne–Newbold small-sample correction and a Student-t with
  `n − 1` df. Sensitivity (report only): `m = h − 1` and `m = 167`. If the estimated variance is
  `<= 0`, log null and mark the run invalid for gating. Missing hours are dropped; require at least
  95% of the grid hours present.
- Per-run dev CIs and p-values are descriptive. The holdout CI is the confirmatory test.
- Correlation is not causation: results say "predicts better on this period", not why.

## 7. Minimum sample sizes

- Holdout (+24 h): at least **5,000 pairs** and **400 hourly origins** (≈ 17 d) present, and every
  station in the cohort with at least 250 pairs. Below that the holdout result is
  `insufficient_data`, not a pass.
- Dev run: at least 3 weekly folds with at least 1,000 pairs each.
- Spike metrics: at least 50 observed spike pairs, else null (section 5).
- Category/spike reporting by station only for stations with at least 250 pairs.

## 8. Multiple comparisons policy

- Dev exploration is free (any number of runs, models, features), but every run is logged in
  MLflow and the report states the **number of dev runs** and that dev CIs are unadjusted.
- At most **3 pre-registered finalists** (`final: true`, section 9) go to the holdout, scored
  **once each**. Candidates, features and parameters are frozen in git before scoring.
- Confidence level for the gate: 95% when one finalist is registered; with `k` finalists
  (`k` = number registered, not number scored) use Bonferroni, level `1 − 0.05/k` (two-sided
  percentiles `0.025/k`, `1 − 0.025/k`). The level is fixed in the preregistration set before the
  holdout is touched and logged as param `ci_level`. Each finalist is compared against two
  baselines; both comparisons use the same level.
- A holdout re-run of the same preregistration id is invalid (a crash re-run with identical
  `git_sha`, `snapshot_sha256` and predictions is allowed and noted).
- A failed finalist is reported as failed. New ideas after the holdout is seen need a new
  holdout (more data), never a re-score.

## 9. Pre-registration

Before the first run that carries an `id`, commit `docs/experiments/prereg/<id>.yaml` (template:
`docs/experiments/prereg_template.yaml`; model and loader:
`src/pm25_forecast/experiments/prereg.py`). Fields: id, ticket, hypothesis, model_family, model,
feature_set, params, horizons (24 mandatory), expected_effect, decision_rule, split, final,
author_agent. `final: true` requires `split: holdout` and is allowed for at most 3 files.
The MLflow tag `prereg` = the id (dev exploration without a file: tag `exploratory`; such runs can
never go to the holdout).

## 10. Promotion gate

A candidate is promoted to shadow only if **all** hold on the holdout:

1. `diff_vs_persistence_24h_ci_high < 0` **and** `diff_vs_smoother_24h_ci_high < 0` (7-day block
   CI at the section 8 level).
2. No station has MAE worse than persistence by more than **5%** at +24 h (stations below the
   section 7 minimum are listed; if more than 10% of the cohort is below it, the gate is
   `insufficient_data`).
3. Serving-lag check passes: a test proves features at origin `t` use only `ts <= t−1h` and
   `available_at <= t`, and that the serving feature code is the shared module.
4. Registry manifest validates: the model loads through `models/serving.load_serving_model`
   (`feature_list`, `assumed_reporting_lag_hours`, training window, `snapshot_sha256`, artifact hash).
5. Valid run (section 12), holdout minimums (section 7) met.

Passing is a shadow-run recommendation, not production promotion; Boss decides promotion.

## 11. MLflow logging

Tags (all required): `ticket`, `agent`, `model_name`, `model_family`, `feature_set`, `git_sha`,
`git_dirty`, `snapshot_sha256`, `split` (`dev` | `holdout`), `prereg`.
Metrics (+24 h, plus `_6h`/`_12h` variants): `mae_24h`, `rmse_24h`, `skill_vs_persistence_24h`,
`diff_vs_smoother_24h`, `diff_vs_smoother_24h_ci_low`, `diff_vs_smoother_24h_ci_high`,
`diff_vs_persistence_24h_ci_low`, `diff_vs_persistence_24h_ci_high`, `dm_pvalue_vs_smoother_24h`,
`spike_recall_24h`, `spike_precision_24h`, `n_pairs_24h`. Params: seed, `ci_level`, `H0`, `E`,
`fold_starts`, bootstrap B and seed. Undefined metrics are omitted, not zero.

## 12. What counts as a valid run

A run is valid only if all hold; otherwise tag it `invalid` and exclude it from results tables:

- all tags and the required metrics present, `snapshot_sha256` equals the frozen snapshot;
- `git_dirty = false` for `split=holdout` (recommended for dev);
- seeds fixed, code reproducible from the repo with `uv`; no hand-typed numbers;
- air4thai-era data and target exclusions of section 3 enforced; serving lag of section 2 enforced;
- walk-forward and fold dates exactly per section 4; no holdout data read in dev runs;
- common-intersection coverage and minimum sample sizes met (sections 1, 7);
- holdout: prereg file committed before the run, `final: true`, at most 3 finalists, first scoring;
- no NaN/inf predictions; negative predictions clipped at 0 and stated.

## 13. Report (Confluence, 8 sections)

1 Summary; 2 Hypothesis and decision rule (from the prereg); 3 Setup (snapshot hash, folds,
refit, features); 4 EDA and data checks; 5 Results table (MLflow run ids); 6 Statistical tests
and diagnostics (CI, DM, sensitivities, station gate); 7 Figures; 8 Conclusion, limitations, next.
