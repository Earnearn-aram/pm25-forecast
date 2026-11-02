# Model lifecycle and promotion policy (KAN-117)

Status: proposed operational target pending owner approval; implementation gaps are identified below. This policy does not
claim a model is currently served from the champion pointer. All timestamps are UTC.

## Roles and serving contract

- **Persistence baseline** carries forward the newest eligible PM2.5 observation for up to three
  hours. If no eligible observation is present, report no forecast for that station; do not invent
  one. Keep the baseline available and attributable as `persistence-24h`.
- **Candidate** is a versioned, immutable model artifact and manifest produced for evaluation.
  Registration alone does not make it champion or change any served forecast.
- **Shadow challenger** runs alongside the serving path for comparison. Its model version is
  recorded with every prediction. Shadow failures must not prevent the baseline or champion from
  being recorded. Shadow outputs are labelled and do not silently become the served forecast.
- **Champion** is the registry pointer to the reviewed model designated for serving. The intended
  inference contract is to resolve and verify that pointer, store the model version with each
  prediction, and make it visible in the dashboard. If no usable champion exists, use the eligible
  persistence baseline and report the fallback; if data is missing, report no forecast.

Current implementation: Predict reads `CHALLENGER_MODEL_VERSION`, not the champion pointer.
KAN-119 implements the target contract. KAN-121 covers presentation and KAN-122 verifies the
complete path. Do not treat a registry pointer update as proof that inference has changed.

## Evidence and eligibility

Use observations and forecast vintages that were available by the simulated forecast issue time;
labels must be mature and available by the training/evaluation cutoff. Keep issue time, valid time,
station, model version and actual availability provenance. Backfilled timestamps alone do not
prove past availability. The current mutable observations table cannot reconstruct overwritten
values; KAN-115/118 track the gap. An insufficient dataset is a not-ready result, not permission
for a weaker time rule.

Compare candidate, persistence and any incumbent on identical `(station_id, issue_time,
valid_time)` keys and report sample sizes, horizon, station coverage, evaluation window, MAE and
missing-data exclusions. Evaluate the actual immutable artifact that would be promoted. Record
training window, dataset/feature provenance, model version and evaluation results.

## Decision and promotion

The weekly retrain currently checks +24 h on at least 100 matched pairs, requires candidate MAE
below persistence and, when an incumbent exists, at least 2% lower MAE than the incumbent. It
rejects a candidate worse than the incumbent by more than 2% at another mutually supported horizon.
These operational checks are minimum evidence, not proof of statistical significance. If a
champion cannot be loaded or scored, do not replace it based on incomplete evidence.

The proposed policy is owner-reviewed manual promotion. The current implementation permits
`AUTO_PROMOTE=true`; it does not enforce that policy. Keep `AUTO_PROMOTE` unset/false pending an
owner decision; no model becomes champion
solely because it was registered or ran in shadow. The reviewer checks matched coverage,
per-station behavior, data readiness, leakage safety, artifact identity and evaluation limitations;
if evidence is insufficient, retain the current champion or baseline. Record the decision and
reason. KAN-120 implements the unified enforcement and audit; existing registry events already
record pointer changes.

Open decisions blocking model promotion: minimum per-station sample and coverage thresholds,
acceptable station-level degradation, the required evaluation period and how to handle overlap
between the incumbent's training data and the evaluation window. Until the owner approves these,
the current 100-pair and 2% checks alone do not authorize promotion under this proposed policy.

The experiment programme's locked holdout and its stricter pre-registered selection gates are
separate from the rolling seven-day operational window. This agent must never score the locked
holdout. Before any operational evaluation, check whether its evaluation data overlaps a locked
experiment holdout; exclude that period or block the run if exclusion cannot be verified. Do not
use holdout results for tuning. KAN-99 controls the planned re-evaluation; coordinate KAN-94
and KAN-104/110 rather than silently replacing their decisions with the weekly threshold.

## Operations and rollback

A promotion should produce a verifiable chain from decision and artifact through registry pointer,
subsequent forecast model version and dashboard. Rollback moves the pointer to a verified prior
version and records the event; operators must then verify a new inference and dashboard output.
Registry rollback alone is not evidence that serving changed until KAN-119 is in place. If no
prior champion is usable, apply the documented persistence fallback and alert. KAN-122 rehearses
promotion, failure, fallback and rollback without production changes.

Monitor job execution, input freshness, forecast coverage and matched model performance separately.
Report unavailable/waiting states honestly. Alerts distinguish operational failures from expected
insufficient-data states; use KAN-114 and KAN-121 for those changes. Never disable leakage or
readiness checks to meet a deadline.
