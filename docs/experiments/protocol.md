# Experiment protocol (KAN-79, story KAN-78)

Single source of truth for every experiment under KAN-78. Fixed by Boss on 2026-11-01. Do not
deviate silently: a deviation makes the run invalid (section 12) unless this file is amended in a
reviewed PR first. Words MUST / MUST NOT are binding.

**Goal.** Find a +24 h PM2.5 model that beats persistence **and** the smoother on a locked
holdout, with a paired CI (level in section 8) entirely below zero, servable through the existing registry and
shadow path.

## 1. Targets, horizons, baselines

- Target: hourly PM2.5 (µg/m³) of a station at `t + h`. Primary horizon **+24 h**; secondary +6 h
  and +12 h (reported, never gating).
- **Persistence**: newest observation with `ts` in `[t−3h, t−1h]` (the 3 h age limit of
  `models/baselines.py`, applied after the serving lag); none, no forecast.
- **Smoother** (KAN-76, merged; `features.mean_24h`, `models/serving.py`):
  `latest + 0.5 · (mean24 − latest)`, `latest` as above, `mean24` the plain mean of usable
  readings with `ts` in `[t−24h, t−1h]` and `available_at <= t`, **no minimum count**. When
  `mean24` is None the smoother falls back to `latest` (= persistence). Baseline scoring uses this
  same code at +6/+12 too, although the registered smoother serves only +24 h. The merged code
  wins over this text on any conflict.
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

**Assumed availability in experiments.** `docs/backtest.md` forbids substituting `ts` for
`available_at` because the live `ingested_at` is not a historical arrival time. For the frozen
experiment snapshot we apply an explicit, documented assumption instead: `available_at = ts + 1 h`
for every air4thai observation (backfilled rows have an `ingested_at` days later, which would
distort walk-forward training). This is the serving lag, not a measured arrival time, and every
report states it. CAMS vintages keep their real `available_at` (collected only since 2026-10-27). The default snapshot therefore has 0 vintages; a CAMS-inclusive snapshot needs `--kan74-repaired` (KAN-80).
KAN-74-affected targets stay excluded until the repair (section 3).

## 3. Data

- Frozen snapshot from E02 (KAN-80), identified by `snapshot_sha256`. Never re-export mid-story.
- **air4thai era only**: `ts >= 2026-07-31T17:00:00Z` (`S0`). Earlier openaq rows MUST NOT enter
  features, labels or training (source break: openaq is ~8.6× smoother hour to hour; KAN-65).
- **Targets at or after 2026-10-26T00:00:00Z are excluded** until the KAN-74 repair; rows after that may
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
  better.** Pooled mean of paired absolute-error differences (section 6).
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
  (KAN-65: spike detection was very weak, ≤ 10% recall; report it honestly. Values > 37.5 are
  likely rare in Aug-Oct, so spike metrics may be null or unstable.)
- Each metric is logged with the `_6h`/`_12h` variants for every horizon in the preregistration.

## 6. Statistics

Unit of analysis for inference (bootstrap **and** DM, one unit everywhere): the **hourly,
pair-weighted series**. For origin hour `t`, `S_t` = sum of the paired absolute-error differences
(candidate − baseline) over the stations present, `N_t` = their count. The estimate is
`sum(S_t)/sum(N_t)`, identical to the pooled pair mean `diff_vs_*_24h`; the DM series is
`d_t = S_t/N_t`. (Stations are not averaged with equal weight: station means would differ,
about 109 vs 116 in the KAN-65 numbers, so use pair weights.) Compute in float64 with no rounding
before the CI. Consecutive origins overlap in time and h-step errors are autocorrelated: pairs
MUST NOT be treated as independent. `skill_vs_persistence_24h` and the spike metrics are pooled
over all pairs of the split.

- **Moving-block bootstrap CI, 72 h primary.** Blocks are 72 consecutive hourly origins on the
  calendar grid, start uniformly at any grid hour that fits; each block carries `(S_t, N_t)`
  sums; draw `ceil(n_hours/72)` blocks with replacement. **B = 10,000 resamples, seed 20261101**,
  `numpy.random.default_rng(seed)`, percentile interval. `diff_vs_*_ci_low/_ci_high` are the
  bounds at the section 8 level (logged as param `ci_level`). Mandatory sensitivities: 24 h and
  168 h blocks. If any sensitivity interval crosses 0 while the 72 h one does not, mark the result
  *fragile* in the report (KAN-65: borderline CIs flipped with block size).
- With a 21 d holdout there are only 3 independent weekly blocks: say so in every holdout report.
- **Diebold-Mariano**, descriptive only (the CI decides). Absolute-error loss, two-sided,
  versus the smoother: `dm_pvalue_vs_smoother_24h`. Long-run variance by Newey-West (Bartlett
  kernel), primary bandwidth **`m = 167`** (one week, covers h-step overlap and the daily cycle);
  sensitivity `m = 47` (`2h − 1` at h = 24) and `m = h − 1`. Harvey-Leybourne-Newbold
  small-sample correction and Student-t with `n − 1` df. If the estimated variance is `<= 0`, log
  null and mark the run invalid for gating. Missing hours are dropped; require at least 95% of
  the grid hours present.
- Per-run dev CIs and p-values are descriptive. The holdout gate (section 10) is **supportive
  evidence on one 3-week window, not a confirmatory test**: a short window, 3 independent
  blocks, one season, and baselines tuned (beta) on earlier data.
- Correlation is not causation: results say "predicts better on this period", not why.

## 7. Minimum sample sizes

- Holdout (+24 h): at least **2,100 pairs** and **400 hourly origins** (≈ 17 d) present, and every
  station in the cohort with at least 250 pairs. Below that the holdout result is
  `insufficient_data`, not a pass. (Maximum with 5 stations: 5 × 21 × 24 = 2,520 pairs.) The main
  coverage gate is the 95% rule of section 1.
- Dev fold: at least **600 pairs** (maximum 5 × 7 × 24 = 840); a dev run needs at least 3 such
  folds. Date arithmetic and these minimums are tested in `tests/test_experiment_folds.py`
  (`src/pm25_forecast/experiments/folds.py`).
- Spike metrics: at least 50 observed spike pairs, else null (section 5).
- Category/spike reporting by station only for stations with at least 250 pairs.

## 8. Multiple comparisons policy

- Dev exploration is free (any number of runs, models, features), but every run is logged in
  MLflow and the report states the **number of dev runs** and that dev CIs are unadjusted.
- At most **2 pre-registered finalists** (`final: true`, section 9) go to the holdout, scored
  **once each**. Candidates, features and parameters are frozen in git before scoring.
- Confidence level for the gate: 95% when one finalist is registered; with `k` finalists
  (`k` = number registered, not number scored) use Bonferroni, level `1 − 0.05/k` (two-sided
  percentiles `0.025/k`, `1 − 0.025/k`; at most 97.5%). The level is fixed in the
  preregistration set before the holdout is touched and logged as param `ci_level`. Each finalist is compared against two
  baselines; both comparisons use the same level.
- Implementation (KAN-82): `finalists` (k, 1 or 2) is a field of the preregistration and `evaluate --holdout` takes k and the horizons only from it. The once-only check queries the local MLflow store (`~/pm25-experiments/mlflow`); a deleted or replaced store resets it, so never delete that store and keep backups. If the store cannot be queried the holdout is refused.
- A holdout re-run of the same preregistration id is invalid (a crash re-run with identical
  `git_sha`, `snapshot_sha256` and predictions is allowed and noted).
- A failed finalist is reported as failed. New ideas after the holdout is seen need a new
  holdout (more data), never a re-score.

## 9. Pre-registration

Before the first run that carries an `id`, commit `docs/experiments/prereg/<id>.yaml` (template:
`docs/experiments/prereg_template.yaml`; model and loader:
`src/pm25_forecast/experiments/prereg.py`). Fields: id, ticket, hypothesis, model_family, model,
feature_set, params, horizons (24 mandatory), expected_effect, decision_rule, split, final,
author_agent. `final: true` requires `split: holdout` and is allowed for at most 2 files.
The MLflow tag `prereg` = the id (dev exploration without a file: tag `exploratory`; such runs can
never go to the holdout).

## 10. Promotion gate

A candidate is promoted to shadow only if **all** hold on the holdout:

1. `diff_vs_persistence_24h_ci_high < 0` **and** `diff_vs_smoother_24h_ci_high < 0` (72 h block
   CI at the section 8 level), **and** the point estimate `diff_vs_smoother_24h <= −0.15` µg/m³
   (minimum effect).
2. The paired diff (candidate − baseline) is negative in **each of the 3 weekly holdout folds**,
   against both persistence and the smoother.
3. No station has MAE worse than persistence by more than **5%** at +24 h. Report a bootstrap CI
   (same blocks) for each station's relative difference; the gate fails on the point estimate;
   the rule is to report the CI and flag every station whose CI includes +5%. Stations below the section 7 minimum are
   listed; if more than 10% of the cohort is below it, the gate is `insufficient_data`.
4. Serving-lag check passes: a test proves features at origin `t` use only `ts <= t−1h` and
   `available_at <= t`, and that the serving feature code is the shared module.
5. Registry manifest validates: the model loads through `models/serving.load_serving_model`
   (`feature_list`, `assumed_reporting_lag_hours`, training window, `snapshot_sha256`, artifact hash).
6. Valid run (section 12), holdout minimums (section 7) met.

Passing is supportive evidence (section 6) and a shadow-run recommendation, not production
promotion; Boss decides promotion.

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
- holdout: prereg file committed before the run, `final: true`, at most 2 finalists, first scoring;
- no NaN/inf predictions; negative predictions clipped at 0 and stated.

## 13. Report (Confluence, 8 sections)

1 Summary; 2 Hypothesis and decision rule (from the prereg); 3 Setup (snapshot hash, folds,
refit, features); 4 EDA and data checks; 5 Results table (MLflow run ids); 6 Statistical tests
and diagnostics (CI, DM, sensitivities, station gate); 7 Figures; 8 Conclusion, limitations, next.
