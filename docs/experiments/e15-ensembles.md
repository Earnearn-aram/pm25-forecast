# E15 Ensembles of the family-best models (KAN-92)

Verdict: **NO CONFIRMED WIN, ONE LEAD.** No ensemble meets the preregistered rule. The stacker is worse than the smoother, and equal and inverse-error averages of all five family-best models are indistinguishable from each other (3.734 and 3.736) with CIs that include zero. The only ensemble whose 72 h CI is entirely below zero against both baselines is the plain blend of the smoother and the E12 LightGBM (mae_24h 3.731, diff vs smoother -0.137 ug/m3, 95% CI [-0.21, -0.03], 99% CI [-0.24, -0.00]), but it is fragile (the 24 h-block CI reaches +0.01), its point estimate is above the -0.15 minimum effect, and it comes from a dev-only, selection-biased pool. Dev split only; the holdout was never touched.

## 1. Question, rule, shortlist (fixed in git before the runs)

Preregs: `docs/experiments/prereg/e15-*.yaml` (5 files, committed with the code in `020c1df` before any logged run; every run is clean, `git_dirty = false`). Question: does combining the best model of each family lower +24 h dev MAE versus the smoother, and which (at most 3) candidates deserve the locked holdout?

Variants registered (K = 5, all counted for the multiple-comparison correction of protocol section 8):

| id | what it combines | fitted weights |
|---|---|---|
| e15-equal5 | plain average of ridge, climatology, GBM, ARIMA, weather-LightGBM | none |
| e15-invmae5 | the same five, weights proportional to 1/MAE on the newest 14 days of out-of-fold predictions | per fold, from matured targets only |
| e15-stack5 | the same five, non-negative least squares with intercept on all out-of-fold predictions | per fold |
| e15-smoother-gbm | 0.5 x smoother + 0.5 x E12 LightGBM (no cross-station group) | none |
| e15-ridge-clim | 0.5 x pooled ridge + 0.5 x 28 d climatology blend (the two best simple members) | none |

Rule (in every prereg): an ensemble counts as better than the smoother only if its 72 h CI at the Bonferroni level 1 - 0.05/5 = 99% is entirely below zero against both persistence and the smoother, the point estimate vs the smoother is <= -0.15 ug/m3, and it is not fragile. Shortlist rule: CI vs persistence entirely below zero, dev MAE within 0.05 of the smoother, point estimate vs the smoother below zero, serving-lag and station gates pass. Up to 3 are listed; at most 2 would be registered (protocol section 8).

## 2. Members, and three deviations from the ticket text

Members are the family-best configurations of E05, E06, E07, E11 and E12, with the frozen parameters of their preregistrations (`COMPONENT_PARAMS` in `src/pm25_forecast/experiments/ensembles.py`; a test compares them with the YAML files). Nothing was re-tuned. Three members differ from the names in the ticket, each for a servability reason found while testing:

1. **GBM = `e12-abl-no-cross`, not `e12-full-l2-delta`.** The full model reads the other stations from a memo of earlier calls. A fresh instance (what the serving-lag probe uses) predicted 14.874 where the live one predicted 14.865 (same origin, same data), so any ensemble containing it could only be `inconclusive` at gate 10.4. The no-cross twin is an E12 ablation that was preregistered and run in E12 (mae_24h 3.874 vs 3.868 for the full model; difference inside noise) and reproduces exactly.
2. **Weather = `e11-lgbm-wx-d2` (previous_day2), not `e11-lgbm-wx` (previous_day1).** The d2 model is better (3.958 vs 3.970) and its weather is strictly available at the origin (protocol section 2); d1 is not (E11 caveat).
3. **ARIMA runs with a `stateless` flag** (new, off by default, in `timeseries.py`): every call filters the last 336 h with the frozen refit parameters instead of extending a state chain. The chain version differs from a fresh instance at about 1e-7, above the probe's 1e-9 tolerance. Forecasts differ from the E07 chain by up to about 4e-4 ug/m3 in the 4 origins compared (not a full-run comparison). While adding the flag I found that the existing fallback branch of `advance` called `.apply` on the chain object instead of on its results (it would raise); fixed with `.results`.

## 3. Method

- **Combination at predict time.** One factory (`pm25_forecast.experiments.ensembles:fit`) per fold: it fits the members on the same `TrainSnapshot`, and the predictor combines their forecasts. It first rebuilds the snapshot with only rows that have `ts <= origin - 1 h` and `available_at <= origin` (by time, not position; several members slice `snapshot.observations` by position, which the probe punishes). A member that returns None makes the ensemble return None, so all variants score the same rows (coverage 100%, 4,200 pairs).
- **Out-of-fold predictions** (invmae and stack only). Inside each outer fold's training part, `gbm.inner_cv_masks` (not re-implemented) gives 3 expanding-window inner splits purged by horizon + 1 h. For every inner split each member is refitted on the rows matured at that split's refit time (the newest training label plus 1 h) and predicts the inner validation pairs from serving-lag snapshots. Nothing from the outer fold and no label unavailable at the validation origin is used (tested: `test_out_of_fold_predictions_respect_the_purge`). Weights are frozen at the weekly refit; they do not update inside the week. ARIMA needs 14 days, so in the early folds its first inner split has no forecast and those rows are dropped (OOF rows per outer fold: 1,695 / 2,115 / 2,535 / 2,955 / 5,065).
- **Inverse-error weights**: 1/MAE of each member over the newest 14 days of OOF rows, normalised. **Stacker**: `LinearRegression(positive=True)` with intercept on all OOF rows, pooled over stations.
- Evaluation: the standard runner (dev, 5 weekly folds, 5 stations, horizon 24 only, 72 h block bootstrap B = 10,000, seed 20261101, DM with Newey-West m = 167). Member side effects (KAN-84 coefficient dump, KAN-88 prediction CSV) are redirected to `~/pm25-experiments/cache/kan92` so the earlier tickets' files are not overwritten.

## 4. Results (dev, +24 h; 4,200 pairs; persistence 4.311, smoother 3.867)

Numbers are printed by `python -m pm25_forecast.experiments.kan92_analyze` from the tracked runs.

| run | mae_24h | skill vs pers. | diff vs pers. [95% CI] | diff vs smoother [95% CI] | folds < 0 vs smoother | DM p (m=167) | fragile | station gate | serving lag | MLflow |
|---|---|---|---|---|---|---|---|---|---|---|
| e15-smoother-gbm | 3.731 | +0.135 | -0.581 [-0.78, -0.40] | -0.137 [-0.21, -0.03] | 5/5 | 0.013 | yes | pass | pass (1200 checks) | 2ac20eb2 |
| e15-equal5 | 3.734 | +0.134 | -0.578 [-0.85, -0.35] | -0.133 [-0.33, +0.10] | 4/5 | 0.218 | no | pass | pass | af5be7e3 |
| e15-invmae5 | 3.736 | +0.133 | -0.575 [-0.85, -0.35] | -0.131 [-0.33, +0.10] | 4/5 | 0.225 | no | pass | pass | 8b86b1ea |
| e15-ridge-clim | 3.741 | +0.132 | -0.570 [-0.83, -0.32] | -0.126 [-0.34, +0.13] | 3/5 | 0.208 | no | pass | pass | 7491fbe5 |
| e15-stack5 | 3.879 | +0.100 | -0.432 [-0.83, -0.10] | +0.012 [-0.33, +0.35] | 3/5 | 0.941 | no | pass | pass | d1f5ac7b |

Members, for reference (earlier tickets): ridge 3.757 (diff vs smoother -0.110 [-0.31, +0.13]), climatology 3.791 (-0.076 [-0.34, +0.21]), E12 GBM 3.868 (+0.001 [-0.23, +0.26]), ARIMA 3.965 (+0.098 [-0.07, +0.23]), weather d2 3.958 (+0.091 [-0.37, +0.47]). The four averaging ensembles are each better than every single member, but equal5 and invmae5 are only 0.02 better than the pooled ridge alone, and the CIs overlap completely.

**Correction for the 5 variants.** The Bonferroni run (K = 5, level 99%, run `9381dea0`, same prereg, tracked in `kan78/e15-smoother-gbm-bonf5`) for the best variant: diff vs smoother -0.137 [-0.245, -0.004], diff vs persistence -0.581 [-0.846, -0.339]. Still entirely below zero at 72 h blocks, by 0.004. The other four variants have unadjusted CIs that already include zero, so the correction cannot help them (a wider interval contains the narrower one).

**Why the smoother-gbm lead is weak.**
- Fragile: block-length sensitivity at 95% is 24 h [-0.31, +0.01], 168 h [-0.21, -0.07]; at 99% the 24 h interval is [-0.37, +0.06]. DM is significant only with the widest bandwidth (p 0.013 at m = 167, 0.058 at m = 47, 0.095 at m = h-1).
- The point estimate -0.137 is above the -0.15 floor of the promotion gate; the preregistered rule (<= -0.15) is therefore not met.
- It is negative in all 5 folds, but folds 0 and 3 by only -0.023 and -0.026; fold 4 contributes -0.372.
- Station gate: all 5 stations are better than persistence (worst -0.107 relative, CI high -0.040).
- The GBM was chosen from 16 E12 runs and the other members from about 60 earlier dev runs; the blend itself was specified by the ticket before any data. The 99% interval corrects only for the 5 E15 variants, not for that upstream selection, so it is optimistic.

**Fitted weights.** invmae: mean weights ridge 0.21, climatology 0.20, GBM 0.20, ARIMA 0.19, weather 0.20 (inner-fold MAEs of the members differ by only 0.2, so the weighting is almost the equal average, which is why the two have the same score). stack: ridge 0.33, climatology 0.22, GBM 0.00, ARIMA 0.02, weather 0.40; weights sum 0.86 to 1.10 and intercepts range -0.35 to +1.53 across folds. The OOF MAE of the stacker is below every member's on training data but it is worse than the smoother outside it (+0.012; worse than the smoother in folds 0 and 3). The weights are unstable between folds and the stacker overfits the inner folds; with 5 stations and about 8 weeks of data a fitted combiner has little to learn beyond "average".

## 5. Gates and probe

Serving-lag probe (gate 10.4): `pass` for all 5 variants and the Bonferroni run, 1,200 checks, 0 mismatches, 0 errors (members fitted fresh three times per fold, forbidden rows added with +1000 and -999 shifts). The probe cannot see the weather files read by the weather member (outside `train`/`snapshot`); the cache is keyed by valid hour and lead (tested in E11: a d2 value for hour u is used at origin t only if u - 48 h <= t), and the smoother-gbm and ridge-clim variants do not use it. Station gate (10.3): `pass` for all variants. Coverage 100%, no invalid reasons. Gates 10.5 (registry manifest) and the shared-module half of 10.4 were not checked here: the ensemble would need a new serving wrapper (see caveats).

## 6. Shortlist (at most 3) and the two that would be registered

1. **e15-smoother-gbm: finalist 1.** Lowest dev MAE of everything tried (3.731), the only candidate with a 72 h CI below zero against both baselines at the 95% and the 99% level, negative in 5/5 folds, both gates pass. Cheap to serve (the smoother already exists; the GBM needs the shared v2 feature module and no cross-station state). Cost of being wrong: it is fragile and below the -0.15 floor, so a holdout pass needs a larger effect than dev showed.
2. **e15-ridge-clim: finalist 2.** Same MAE class as the five-member averages (3.741) with only two members that serve from own-station history (the pooled ridge on `fs_v2_core`, same-hour climatology), no weather feed, no ARIMA, no GBM. It is the cheapest ensemble to serve and is what the 5-member averages reduce to (equal5 is 0.007 better, inside noise). Its dev CI includes zero (95% [-0.34, +0.13]), so it is registered as the simple alternative, not as a win.
3. **e06-ridge-pool-raw-chg alone (not registered; control).** Best single model (3.757, -0.110 [-0.31, +0.13], gates and probe pass, 3,275 probe checks in E06). It answers whether blending adds anything beyond the ridge; the dev numbers say it adds at most 0.02. It would only be scored if Boss wants a third look.

Not shortlisted: equal5 and invmae5 (no better than ridge-clim by a margin that matters, and they need live weather forecasts, ARIMA and GBM; E11 already said the weather ingestion ticket is not justified), stack5 (worse than the smoother), and every single non-ridge member (the smoother is at least as good).

If the holdout is used, `finalists` = 2 sets the level to 97.5% (`1 - 0.05/2`); registering only smoother-gbm would keep 95%. Passing would still be supportive evidence on one 3-week window, not confirmation (protocol section 6).

## 7. Caveats

- All numbers are dev, unadjusted except where stated; selection of members (about 80 earlier dev runs) is not corrected. The holdout was not read.
- Weights are fitted on one season (about 8 to 11 weeks) and 5 stations; the stacker result may differ with more data.
- Weights and the stacker are frozen at the weekly refit; a version that updates weights hourly from newly matured targets was not tried (it would need predictions of past origins from models fitted before them).
- Serving: no production wrapper exists for a multi-model blend; `models/serving.load_serving_model` and the registry manifest (gate 10.5) would need an ensemble artefact.
- The smoother-gbm gain is an average of two models that tie individually; I did not test why (error correlation, variance reduction) and make no causal claim.

## 8. Reproduce

Code: `src/pm25_forecast/experiments/ensembles.py` (factory), `kan92_analyze.py` (tables), `tests/test_experiment_ensembles.py`. Snapshot `e02-v1`. Seed 20261101. For each id in `e15-smoother-gbm e15-ridge-clim e15-equal5 e15-invmae5 e15-stack5`:

    uv run --group experiments python -m pm25_forecast.experiments.evaluate \
        --prereg docs/experiments/prereg/<id>.yaml --snapshot ~/pm25-experiments/snapshots/e02-v1 \
        --model pm25_forecast.experiments.ensembles:fit

Bonferroni run: the same command for `e15-smoother-gbm` with `--finalists 5 --run-name e15-smoother-gbm-bonf5 --experiment kan78/e15-smoother-gbm-bonf5`. Tables: `uv run --group experiments python -m pm25_forecast.experiments.kan92_analyze`. Total tracked dev runs for this ticket: 6 (5 variants, 1 Bonferroni). Before the logged runs, a pipeline check (no scoring, no choice made from it) timed each member and compared a fresh instance with the live one on a handful of origins; it is what exposed the GBM memo and the ARIMA chain.
