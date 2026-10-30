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

---

# Amendment 1 and 2 (written before any model was fitted or scored)

State of knowledge at amendment time: data availability by month (row counts per station-month),
the source column (openaq to 2026-07-31 16:00 UTC, air4thai from 2026-07-31 17:00 UTC), and a
sanity check that the persistence MAE on the KAN-15 test pair set at +24 h reproduces 4.4976 with
10,729 pairs. No candidate model has been fitted. This amendment supersedes the "Split",
"Models and search" and "Multiplicity" sections wherever they conflict.

## Amendment 1: global multi-horizon model (coordinator request)

Extra family: ONE LightGBM trained on samples stacked over horizons 6/12/24 with numeric `horizon`
as a feature, in two variants reported separately: G (no station id) and G+id (station id as a
LightGBM categorical). Hypothesis: sharing structure across horizons/stations helps small data, but
with near-univariate inputs the gain is negligible and horizon interactions may hurt +6 h.
Features: set D (+ horizon, and station id only in G+id; station id is also removed from the
per-horizon sets for the "no id" variants: per-horizon sets B-E include station id as in the table,
the global G variant excludes it). Evaluated per horizon against persistence AND the current
per-horizon boosters on the same station-hours. Serving constraint to check (not change):
`models/serving.py` accepts a multi-horizon bundle of one booster per horizon, or a single-horizon
bundle, and checks every booster's feature names equal the manifest `feature_list`, and every name
must be produced by `features.build_features` (only pm25_latest, pm25_same_hour_24h, cams_pm25 now).

## Amendment 2: walk-forward backtest

Harness note: `backtest.py` (KAN-14) is a JSON-input, CAMS-vs-challenger scorer with per-origin
callbacks, 28-day windows and `available_at` gating; it cannot refit models per fold and CAMS
history does not exist. I reuse its rules, not its code: common station-hour intersection per
horizon, no filling of gaps, day-block bootstrap on target days, seeded, no unpaired comparison.
Deviation: the common pair set is "target present and persistence present" (the KAN-15/64 rule),
not the intersection with seasonal naive; seasonal naive is compared to persistence on its own
intersection (its own pairs are a subset), because requiring it would change the headline pair set.

**Folds** (expanding window; train rows need target < boundary B, so a gap of one horizon is left;
test rows are origins in [B, B + window)); boundaries and windows fixed here:

| Run | Train targets before | Test origins | Note |
|---|---|---|---|
| F1 | 2025-12-31 00:00 | 2026-01-01 .. 2026-02-11 (42 d) | dry/haze season |
| F2 | 2026-03-31 00:00 | 2026-04-01 .. 2026-05-12 | bkp gap 05-19..21 is after window |
| F3 | 2026-05-31 00:00 | 2026-06-01 .. 2026-07-12 | export has almost no data 2026-07-03..31, so effective ~33 d |
| F4 | 2026-07-31 00:00 | 2026-08-01 .. 2026-09-11 | air4thai source |
| F5 | 2026-09-11 00:00 | 2026-09-12 .. 2026-10-27 | air4thai source |
| HOLDOUT | KAN-15 test_start (2026-07-10 10:00) | origins 2026-07-10 10:00 .. 2026-10-27 21:00 | headline, reported separately |

F4/F5 test origins lie inside the HOLDOUT span, so folds and holdout are not independent. Tuning
(feature set, target, objective, num_leaves, ridge alpha, and number of rounds) happens only inside
each run's training period: validation slice = last 45 days of that run's training origins
(models fit on origins with target before validation start, rounds by early stopping with patience
30 and cap 600, then refit on full train with the best iteration; ridge/linear has no rounds).

**Candidates, each at +6, +12, +24 h on identical station-hours per run:**
1 persistence (reference); 2 seasonal naive (reading at target-24 h if servable, i.e. horizon
< 24; at +24 h the servable form is target-48 h); 3 current per-horizon LightGBM (set A, direct,
KAN-15 params, 200 rounds, untuned); 4 LightGBM direct, val-selected over set x config; 5 LightGBM
residual (y minus persistence), val-selected; 6 ridge, val-selected over set x target x alpha;
7 global G; 8 global G+id (global selected over objective x leaves x target on pooled-horizon val
MAE). "Best candidate" at each run/horizon = lowest validation MAE among 4, 5, 6 (8 for global
variants shown separately); it is determined without looking at test.

**Grid replaced by a smaller one** (early stopping replaces rounds/learning rate): learning rate 0.05,
objective {l2, l1} x num_leaves {7, 31}, min_child_samples 50, feature_fraction 0.9, deterministic,
seed 20261029. Per run and horizon: 5 sets x 2 targets x 4 configs = 40 LightGBM fits + 40 ridge
(5 sets x 2 targets x 4 alphas). Global: 2 objectives x 2 leaves x 2 targets x 2 variants = 16 fits
per run. Total tuned configs: (80 x 3 + 16) x 6 runs = 1,536, plus 3 untuned current-model fits
per run. Set C additionally includes `pm25_t24` (reading at target-24 h, always NaN at +24 h by
the as-of rule, informative at +6/+12 h). Features are built per horizon with the same reference
time r = origin - 1 h.

**Reporting**: per run and pooled-over-folds (F1-F5; target days resampled jointly in one pool) and
HOLDOUT, per horizon: MAE (primary), RMSE, mean signed error (bias), MAE on high hours (target > 37.5),
MAE in three bands (<= 15, 15-37.5, > 37.5), and for the headline +24 h candidate also per station,
per local hour-of-day group (0-5, 6-11, 12-17, 18-23, Bangkok time) and per target month.
Paired difference vs persistence and vs current LightGBM, day-block bootstrap 95% CI (10,000
draws) per run and pooled; stability = number of folds (of 5) where the candidate's point MAE beats
persistence. Source sensitivity: the holdout split by target source (openaq / air4thai).

**Multiple comparisons**: claims "beats persistence" are the pooled-fold and HOLDOUT comparisons:
7 candidates (2-8) x 3 horizons x 2 = 42 claims. Correction: Bonferroni, i.e. CI at level
1 - 0.05/42 (20,000 draws) must be entirely below zero; Holm-adjusted bootstrap p-values are also
reported. Per-fold comparisons (7 x 3 x 5 = 105) are descriptive only. Total comparisons
disclosed: 42 + 105 = 147 candidate x horizon x run comparisons vs persistence.

**Promotion rule (final, supersedes the earlier one-sentence rule)**: a candidate may be recommended
only if at +24 h ALL hold: (a) HOLDOUT CI entirely below zero after Bonferroni correction;
(b) pooled-fold CI entirely below zero after Bonferroni correction; (c) point MAE beats persistence
in at least 4 of 5 folds; (d) point-estimate paired difference <= 0 in both high-pollution subsets
(target > 37.5 and latest reading > 37.5), pooled and in HOLDOUT. Otherwise: "no reliable gain"
(or "more data needed" when the CI cannot exclude a 0.25 ug/m3 gain).

**Leakage audit**: (i) an independent scalar re-implementation recomputes every feature for 300
sampled rows per horizon from a grid truncated to indices <= origin - 1 and must match the vectorized
table exactly; (ii) a deliberately leaky control (set D + the neighbor stations' mean reading AT the
target hour) must beat persistence massively, demonstrating the harness would expose leakage. The control is
not a candidate and is not counted; (iii) a table of each feature's as-of rule.

**Entry point**: `PYTHONPATH=src python -m pm25_forecast.experiments.run_all --database
data/history.duckdb --out out/kan65` reruns EDA, audit, all runs; prints SHA-256, library versions
and total runtime. Priority if budget runs short: folds and per-horizon comparison, then the global
model, then breakdowns; any skipped item is stated in the report.

---

# Amendment 3: regime break found in EDA (written before any model was fitted)

EDA (script `experiments/eda.py`, output `out/kan65/eda.json`) shows a structural break at
2026-07-31 17:00 UTC, when the `source` column switches from `openaq` to `air4thai` at all five
stations (this is also why the export has a ~28-day hole 2026-07-02..07-29 at every station):

- openaq era (regime A, 2024-10..2026-07-02): very smooth (mean |1 h change| 0.27-0.41 ug/m3,
  10-15% identical consecutive values, hour-of-day profile flat at 24-25 ug/m3, lag-1 ACF 0.9965).
  Looks like a moving-average product, not raw hourly readings (hypothesis, not verified).
- air4thai era (regime B, 2026-07-29 onward): rough (mean |1 h change| 1.3 ug/m3 at 3 stations,
  3.2-4.3 at the bkp stations, no identical consecutive values), real diurnal profile, mean level
  13 vs 24 ug/m3, and only ~0.1% of hours above 37.5 (regime A: 6-26% by station).
- Consequence: the KAN-15 split trains on regime A (52,563 rows) and tests on regime B (all but
  the first ~2 test days). The KAN-15 headline is therefore a cross-source transfer test; hour-of-day
  and smoothing features cannot be learned from regime A (flat profile) but may matter in regime B.
  High-pollution hours (> 37.5) essentially do not exist in the test span, so the high-pollution
  safeguard can only be assessed on regime A folds.

Changes to the plan (everything else stands):

1. Runs: F1, F2, F3 (regime A to A), F4 (train regime A, test 2026-08-01..09-11, regime B), HOLDOUT
   (KAN-15, cross-regime), plus three regime-B-native expanding folds R1, R2, R3: train targets
   before 2026-09-01 / 2026-09-22 / 2026-10-13 00:00 UTC (a one-horizon gap is left), test origins
   2026-09-01..09-21, 09-22..10-12, 10-13..10-27. F5 is dropped (superseded by R1-R3). R runs
   choose the training-data variant on the validation slice: {regime-B rows only, regime A+B with a
   `src_air4thai` 0/1 flag (flag = 1 at test, as live)}; validation slice for R runs = last 10 days of
   the training origins (B data is only ~30-75 days); for A/F runs validation = last 45 days as before.
2. Extra target formulations: direct y, residual y - latest, log-ratio log1p(y) - log1p(latest)
   (prediction = expm1(log1p(latest) + output)). Selection over targets on validation MAE.
3. Feature sets reduced to A, B, D, E (C is contained in D). Grid per (set, target): objective
   {l2, l1} x num_leaves {7, 31}, lr 0.05, early stopping on validation, cap 400 rounds. Ridge:
   alpha {0.1, 1, 10, 100}.
4. Extra simple/robust candidates: S = latest-persistence replaced by a smoother, chosen on
   validation among {latest, mean3, mean6, mean24, latest + beta*(mean24 - latest) with beta in
   {0.1,...,0.9}} (15 variants); ENS = mean of the val-selected LightGBM-residual and ridge
   predictions; SHR = persistence + w*(LightGBM-resid - persistence), w in {0.25, 0.5, 0.75} chosen
   on validation. Per-station models, per-hour-bucket models, ExtraTrees/CatBoost/XGBoost: not
   run (no library installed except lightgbm/numpy/scipy for the first; data too small per
   station/bucket in regime B: ~2k pairs per station; stated as dead ends by design, not tried).
5. Candidates for claims (vs persistence): seasonal naive, current LGBM, S, ridge, LGBM direct,
   LGBM residual (incl. log-ratio), ENS, SHR, G, G+id = 10, over 3 horizons, in 3 reporting groups:
   HOLDOUT, pooled regime-B folds (F4, R1, R2, R3), pooled regime-A folds (F1, F2, F3) = 90 claims.
   Bonferroni at 0.05/90 (50,000 bootstrap draws) and Holm p-values. Per-run comparisons
   (10 x 3 x 7 = 210) are descriptive. Total comparisons vs persistence: 300.
6. Final promotion rule at +24 h: (a) HOLDOUT corrected CI entirely below zero; (b) pooled
   regime-B folds corrected CI entirely below zero; (c) point MAE below persistence in at least 3
   of the 4 regime-B folds; (d) no deterioration (point difference <= 0) in the high-pollution
   subset (target > 37.5), measured in the pooled regime-A folds (the only place with enough
   high hours), and, where n >= 30, in the regime-B pools. All four needed. This replaces the earlier
   rule (b-d of Amendment 2).
7. Number of fits is counted by the script and printed; the report quotes that number.
8. Data proposals (not fetched): weather (wind, boundary-layer height, rain), fire hotspots (FIRMS),
   traffic, and the raw hourly Air4Thai history before 2026-07 to remove the regime break; see the
   report.

---

# Amendment 4: primary candidate made explicit; transparent log of a first run

Written after a first complete run (log in this history: commit "walk-forward harness") whose
candidate list lacked the "val-selected best" candidate promised in Amendment 2. Honest disclosure
of what was seen before this amendment: per-family results of that first run (all 10 candidates,
all runs). The change below does not alter any model, split, grid or seed; it only adds the
candidate that Amendment 2 already defined as "best candidate" and recomputes the multiplicity.

1. New candidate `val_best` = the family with the lowest validation MAE among {LightGBM direct,
   LightGBM residual (incl. log-ratio), ridge} for each run and horizon (validation MAE only; no
   test information). It is the single primary candidate for the promotion rule (Amendment 3 item 6).
2. Claims: 11 candidates x 3 horizons x 3 groups = 99 (Bonferroni 0.05/99, Holm over 99).
   Per-run comparisons: 11 x 3 x 7 runs = 231. Total comparisons vs persistence: 330.
3. Actual tuning counts are reported from the script counters (early-stopped LightGBM fits, ridge
   fits, refits). The "1,536" figure in Amendment 2 was a pre-Amendment-3 estimate and is superseded.
4. The high-pollution safeguard (rule d) is applied exactly as written in Amendment 3, in both
   forms (conditioning on the realised target > 37.5, and on the forecast-time-observable latest
   reading > 37.5). The report will state the outcome even if it fails and will separately discuss that
   conditioning on the realised outcome penalises any shrinking predictor mechanically; that
   discussion does not change the pre-registered verdict.

---

# Amendment 5: redesign after independent review (WRITTEN AFTER SEEING RESULTS)

Honesty statement: this amendment is written after the v1 results (PR #51, report v1) were seen
and after an independent reviewer reproduced them and found design problems. Anything below that
was chosen with knowledge of v1 results is flagged "informed by v1". The v1 harness and its outputs
stay in the repository (`run_all.py`, `out/kan65/`) for transparency; v2 is a separate entry point
(`run_review.py`, `out/kan65_v2/`). Nothing is tuned on a test span.

1. **No overlapping test windows.** v1 folds F4 and R1 overlapped (1-11 Sep) and F4/R1-R3 lay inside
   the HOLDOUT span. v2 regime-B folds are consecutive, non-overlapping, regime-B-native windows:
   G1 train targets < 2026-08-22, test origins 08-22..09-11; G2 train < 09-12, test 09-12..10-02;
   G3 train < 10-03, test 10-03..end (10-27 21:00). Regime-A folds F1-F3 are unchanged (they do not
   overlap). F4 and R1-R3 are dropped. The script asserts that no (station, origin) is tested twice
   within a pooled group and reports unique station-hour and day counts. The KAN-15 HOLDOUT
   (origins 07-10 10:00 to the end) is kept as the second headline group but it **overlaps** G1-G3
   in time, so HOLDOUT and the regime-B pool are NOT independent evidence; they are two views of the
   same ~90 days of regime B (the HOLDOUT additionally tests training on regime A only).
2. **7-day-block bootstrap** (blocks of 7 calendar target days, seed 20261030, 40,000 draws for family
   claims) is run alongside the 1-day block for every headline CI. The decision uses the 7-day
   block (the more conservative, as consecutive days are dependent); the 1-day CI is shown for
   comparability with KAN-64. Per-run CIs (3 weeks) use 1-day blocks only and are descriptive.
3. **High-pollution rule (d) replaced.** Selecting hours by the realised target (> 37.5) conditions on
   the outcome and mechanically penalises any shrinking predictor; v1 rule (d) is withdrawn.
   New rule uses forecast-time information: subset = hours whose latest reading (= the persistence
   forecast) is > 37.5. Testable only if the subset has >= 200 station-hours and >= 10 distinct target
   days in the group; else the result is "untestable" (not pass, not fail). If testable: FAIL if the
   7-day-block 95% CI of (candidate minus persistence) MAE lies entirely above zero, otherwise PASS
   with the CI shown. Event-level metric, defined now without looking at onsets by group: an
   **onset** at horizon h is a station-hour whose realised target exceeds the latest reading by
   >= 15 ug/m3; a candidate **catches** it if its forecast exceeds the latest reading by >= 7.5 ug/m3
   (half the event size). Reported per group: number of onsets (and distinct days), share caught
   (7-day-block CI), false-alarm share among non-onsets (forecast rise >= 7.5), mean error and MAE at
   onsets vs persistence. Persistence catches none by construction. If onsets < 30 in a group the
   event metric is "untestable". With ~60 regime-B test days the safeguard may be untestable
   there; if so, the report says so and the recommendation is capped at "shadow only".
4. **Selection uses same-regime validation.** ML models in v2 use FIXED pre-declared configurations
   (no tuning at all, see 5), so the only selection left is the smoother parameter beta:
   regime-B folds G1-G3 choose beta on the last 10 days of their regime-B training data; regime-A
   folds choose beta on the last 45 days of regime-A training data. The HOLDOUT has **no
   regime-B validation data before its test start** (regime B begins 2026-07-29), so selection is
   impossible there: beta is fixed at 0.5 (a priori midpoint; informed by v1 where chosen betas
   were 0.3-0.7) and ML models use their fixed configurations. The regime-B training variant is
   fixed: regime A+B rows with a `src_air4thai` flag (flag=1 as live); the B-only variant is shown as a
   descriptive sensitivity.
5. **Small pre-declared candidate set, fixed grids.** Candidates: persistence (reference); smoother
   (latest + beta*(mean24 - latest), beta in {0, 0.1, ..., 0.9}, 0 = persistence); LightGBM direct
   (feature set D, l1 objective, 7 leaves, lr 0.05, 150 rounds, min_child_samples 50, feature_fraction
   0.9, seed 20261029); LightGBM log-ratio (same config, target log1p(y)-log1p(latest)); ridge (set D,
   residual target, alpha 10, closed form); current per-horizon LightGBM (KAN-15 params; reference
   from KAN-64). Descriptive extras outside the family: smoother with fixed beta 0.5, the beta
   curve, B-only variants, global multi-horizon LightGBM with and without station id (fixed config).
   **Family for correction:** 5 candidates (smoother, LGBM direct, LGBM log-ratio, ridge, current)
   x 3 horizons x 2 headline groups (HOLDOUT, unique regime-B pool) = 30 claims; Holm and Bonferroni
   (0.05/30) are both reported, separately for the 1-day and 7-day block bootstrap. Regime-A pool is
   descriptive and used for the safeguard. Number of fits is printed by the script; there is no hyper-
   parameter search besides the beta grid (10 values per run and horizon).
6. **Smoother evaluated prominently**: beta choice procedure as in 4; table of chosen beta and the
   test-optimal beta per fold (descriptive only), full beta curve of the paired difference per
   group, per-fold differences, folds beating persistence, fixed-beta-0.5 version, and the
   high-pollution and onset results for it.
7. **Decision rule (v2), for the single headline candidate named in advance as the smoother** (selected
   a priori on being the lowest-risk, near-parameter-free candidate, informed by v1), and the
   same rule applied to every other family member for information: at +24 h (a) HOLDOUT 7-day-block
   Bonferroni(30) CI entirely below zero; (b) unique regime-B pool 7-day-block Bonferroni(30) CI
   entirely below zero; (c) point MAE below persistence in at least 2 of the 3 regime-B folds;
   (d) safeguard as in 3 not FAIL in the regime-A pool and, if testable, in the regime-B pool.
   Verdict states: PROMOTE only if a-d pass AND the safeguard is testable in regime B; "SHADOW"
   if a-c pass but the regime-B safeguard is untestable; otherwise no gain / more data. This rule is
   shadow-or-promote advice only: nothing is implemented in this PR.

## Amendment 5b: regime-break hypothesis H-MA (added before the v2 run; coordinator request)

Facts noted: the source switch (2026-07-31 17:00 UTC) equals the earliest hour that the Air4Thai
history endpoint served when probed (docs/kan-9-history-findings.md, rolling window), so the break
may be an artifact of data assembly (source precedence), and OpenAQ's provider for these stations is
Air4Thai itself, whose public app displays 24 h averages. **H-MA:** the openaq-era series is
Air4Thai's trailing 24 h average, not hourly values. Test (EDA only, no model):
- Series S_raw = air4thai-era hourly values; S_ma = trailing 24 h mean of S_raw per station
  (window of the 24 hours ending at t, needs >= 18 present values); S_old = openaq-era values.
- Statistics per series, pooled over stations: mean |1 h change|, share of identical consecutive
  values after rounding to 0.1, lag-1 and lag-24 autocorrelation, diurnal flatness
  (max minus min of the mean by local hour, divided by the mean).
- Season control: compare S_old restricted to Aug-Oct 2025 against S_raw and S_ma (Aug-Oct 2026),
  both on mean level and on the statistics above.
- Decision declared now: H-MA **supported** if S_ma matches S_old(Aug-Oct 2025) within a factor of 1.5
  on mean |1 h change| and diurnal flatness and both have lag-1 ACF > 0.99, while S_raw differs from
  S_old by more than a factor of 2 on mean |1 h change|; **rejected** if S_ma differs from S_old by
  more than a factor of 2 on mean |1 h change| or S_old shows a diurnal amplitude clearly like S_raw;
  otherwise **undecided**. Also reported: whether the level difference (about 24 vs 13 ug/m3) shrinks
  under the season control. Implication for the target (stated in the report): if H-MA holds, regime
  A predicts a different quantity (a daily average) from the live hourly value, so models trained on it
  are not trained on the live target.
