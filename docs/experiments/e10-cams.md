# E10: CAMS forecast as a feature (KAN-87)

**Verdict: NO-GO for now. The question cannot be evaluated on the frozen snapshot: CAMS coverage of the dev split is 0%, structurally, not by bad luck.**

## Question and rule (fixed before running)

Question: do CAMS features lower +24 h MAE of a linear model and of LightGBM, compared with the same
model without them on identical rows? Preregistrations: `docs/experiments/prereg/e10-{ridge,lgbm}-{base,cams}.yaml`
(committed in b1a6b27 before the dry run below). Rule: report coverage first; evaluate only if the matched rows
meet protocol section 7 (3 dev folds of at least 600 pairs); otherwise "not evaluable".

## Features (`src/pm25_forecast/features/cams.py`, shared with serving)

- `cams_pm25_target`: newest vintage (max `issue_time`) for the target hour with `issue_time <= t` and `available_at <= t`.
- `cams_minus_latest`: that value minus the newest own reading in `[t-3h, t-1h]`.
- `cams_err_24h`, `cams_err_72h`: mean `observed - CAMS` over the last 24 / 72 h, matured targets only
  (observation with `available_at <= t`; CAMS is the newest visible vintage issued at or before that hour, so it is a forecast, not a hindcast;
  at least 6 matched hours, else None).
Missing means None, nothing imputed. Unit tests: `tests/test_features_cams.py` (visibility, hindcast exclusion, maturity, matched rows).

## Coverage (`python -m pm25_forecast.experiments.kan87_coverage`, dev split, snapshot e02-v1)

| | |
|---|---|
| +24 h dev origins (5 stations, 5 folds) | 4,200 |
| with a usable CAMS vintage for the target hour | 0 (0.0%) |
| with all four CAMS features | 0 (0.0%) |
| vintages in the snapshot | 0 |

Why it cannot improve with a re-export: dev targets are before H0 = 2026-10-05. The collector stores
vintages only since 2026-10-27, and the runner drops vintages with `valid_time >= H0` (and all targets from 2026-10-26 are excluded
until the KAN-74 repair). A `--kan74-repaired` snapshot would still have 0% dev coverage; the holdout (targets up to 2026-10-25) is also before the first vintage
and is locked anyway. The covered period and the air4thai dev period do not overlap at all.

A dry run (`--no-log`, `ridge_cams` through the runner) ended `INVALID: no +24h pairs`. No MLflow runs were made, so there are no with/without MAE numbers, no CI versus the matched no-CAMS model or the smoother. I did not substitute anything for them.

## Recommendation (confidence: high that it is not evaluable now; none about the effect)

No-go for now; this says nothing about whether CAMS helps. When it can be tested: CAMS-covered days must supply
training rows (28 d) plus at least 3 dev folds, about 49 days from 2026-10-27, so about mid-December 2026 at the earliest, and later for a
holdout (70 d, about early January 2027). That needs a new snapshot with CAMS vintages, a protocol amendment (folds are derived
from S0 and the air4thai era; the matched-rows set will be only the CAMS-covered period), a new prereg set and a new holdout. The factories
(`kan87_models.py`: `ridge_base/cams`, `lgbm_base/cams`) are ready; with that data they run through the standard runner
(`--prereg ... --model pm25_forecast.experiments.kan87_models:<name>`).
Do not read the 0% as evidence against CAMS. Also unresolved: CAMS global is coarse (about 40 km) versus point stations, so a bias-only gain is plausible but unmeasured.

Reproduce coverage: `.venv/bin/python -m pm25_forecast.experiments.kan87_coverage --snapshot ~/pm25-experiments/snapshots/e02-v1`.
