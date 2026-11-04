# Weekly retrain (KAN-23)

## Temporary safety block (KAN-126)

Scheduled/manual retraining and `--if-drift` currently stop before opening a database connection,
exporting observations, building samples, reading the model registry, training, comparing, or
promoting. The result status is `blocked_locked_holdout_boundary_unverified`; the CLI exits 1 and
attempts one reason-specific alert. After that alert is logged/sent, the workflow suppresses its
generic duplicate; if notification fails, the generic alert is the fallback. `AUTO_PROMOTE` does
not bypass this block. Prediction, Score, dashboard reads, and serving the existing champion are
unaffected.

After the holdout-boundary gate is lifted, `AUTO_PROMOTE=true` is still blocked before data or
registry access until an owner-approved automated-promotion policy is implemented. Evaluation remains
recommend-only; a reviewed operator must use the manual registry command to promote.

This is a temporary reduction in retraining functionality. The system has no trusted, versioned
provenance source that binds an operational export to the experiment holdout boundary. Do not
provide an operator-entered date or environment override as a substitute. Retraining can resume
only after a separately reviewed implementation establishes an authoritative, atomic boundary and
provenance source, with tests for revisions and concurrent boundary transitions.

After the KAN-126 safety block is removed, if leakage-safe training data is insufficient, the run is skipped without registering or
promoting a candidate. The summary reports the cutoff and per-horizon counts, and one readiness
alert is sent. Other training errors remain failures. The availability checks are unchanged.

The `retrain` workflow is scheduled every Monday 03:17 UTC (see `retrain-workflow.yml.txt`) and
also supports on-demand runs. Both currently stop at the KAN-126 safety block described above.

## Intended workflow after the safety block is lifted

1. Export the latest hosted history to DuckDB.
2. Train a candidate, version `retrain-YYYYMMDD` (UTC date; `-2`, `-3` on a same-day re-run). The
   training data stops at the start of a held-out window (the last 7 days, `--holdout-days`).
3. Register the candidate (`publish_model`). Registering never changes production.
4. On the held-out window, score the candidate, the current champion, the current challenger
   (`CHALLENGER_MODEL_VERSION`, if set) and `persistence-24h` on identical
   (station, issue_time, valid_time) pairs: MAE and RMSE at +6, +12 and +24 h.
5. `decide()` in `retrain.py` returns promote or keep. The `RUN_SUMMARY` log line holds the full
   table and the decision.

## The model we evaluate is the model we promote

There is no refit on full history. The candidate is trained only on data known before the held-out
window (rows dated before it and ingested before it), so the registered artifact is exactly the one
scored. The price: a promoted model has not seen the last 7 days. A champion or challenger whose
training window reaches into the held-out window is flagged "overlaps holdout" in the RUN_SUMMARY
and the alert, because its scores may be optimistic. The RUN_SUMMARY also logs the candidate
artifact size (`candidate_artifact_bytes`); old models are cleaned up with `registry_cli prune`
(below).

## The rule

Promote only if, at +24 h, the candidate has at least 100 matched pairs, its MAE is at least 2%
lower (relative) than the champion's, and it beats persistence. With no champion, only the pairs
and persistence checks apply. Anything else is keep. The decision uses only the candidate,
champion and persistence pairs; the challenger is reported separately on its own pairs. The
candidate is also kept out if it is worse than the champion by more than 2% relative MAE at any
other horizon both serve (+6/+12 h). A champion that cannot be loaded or scored is never replaced
(the run fails and alerts).

Retraining is recommend-only. `AUTO_PROMOTE=true` returns
`blocked_owner_review_required` before export or registry access. Manual promotion is also blocked
while KAN-117's open policy decisions remain unresolved. The owner must first approve the
per-station sample and coverage thresholds, acceptable station-level degradation, evaluation
period, and overlap handling. After those decisions are approved and implemented, the owner can
review matched coverage, metrics, provenance and artifact identity, then explicitly authorize an
operator to use the manual registry command below.

## Promote manually

```
python -m pm25_forecast.registry_cli check retrain-20261102
python -m pm25_forecast.registry_cli promote retrain-20261102 --confirm
```

Run `check` only after the policy preconditions above are met. Verify the registered model version
and artifact identity against the reviewed evaluation record before authorizing `promote`.

`DATABASE_URL` comes from the environment or `.env`. `champion` shows the current pointer.

## The promotion log (`model_events`)

Every champion change and every retrain decision is one row in `model_events` (migration
`005_model_events.sql`): `id` (the order of events), `at` (the database clock, UTC),
`action` (`promote`, `rollback`, `recommend`, `keep`),
`model_version`, `previous_version`, `actor` (`cli`, `retrain` or `auto`), `metrics` (JSON, the
comparison numbers) and `reason`. A `promote` or `rollback` row is written in the same transaction
as the champion pointer change, so the log and the pointer cannot disagree. The pointer row is
locked (`SELECT ... FOR UPDATE`) before the previous champion is read, so two concurrent promotes or
rollbacks are serialised. Retrain writes `recommend` (and alerts) or `keep`; it cannot write an
automatic `promote`. The manual registry command writes a `promote` row (actor `cli`) atomically
with the pointer.

```sql
SELECT at, action, model_version, previous_version, actor, reason
FROM model_events ORDER BY id DESC LIMIT 20;
```

## Roll back

```
python -m pm25_forecast.registry_cli rollback --confirm
```

Moves the pointer back to the previous champion recorded in `model_events` (the `previous_version`
of the latest `promote` or `rollback`) and logs a `rollback` row (actor `cli`). It refuses, and
changes nothing, if there is no champion, no previous champion is recorded (the very first
promote), that model cannot be served (same check as `promote`), or the latest logged change is not
the live champion (the pointer was moved by hand outside the log; use `promote <version>`).
A second `rollback` goes back to the version you just left, so use `promote <version>` to pick an
older one. Automatic promotion is currently blocked.

## Drift-triggered retrain

`python -m pm25_forecast.retrain --if-drift` currently receives the same KAN-126 safety block as
normal retraining and does not query drift or connect to the database. After trusted boundary
provenance is implemented and reviewed, it is intended to skip without training unless the latest
drift run (`drift.latest_drift_alert`, at most 36 h old) has an alert for
`CHALLENGER_MODEL_VERSION` or the production champion (both are checked).

Cooldown: after the safety block is lifted, it also skips if a retrain decision (any `model_events` row by actor `retrain` or `auto`)
was logged within the last 7 days (`RETRAIN_MIN_GAP`), so persistent drift, a "keep" result or a
same-day re-run cannot retrain every day. When it does run it is
the normal retrain: the candidate is registered and only recommended. The `drift-retrain` job in
`drift-workflow.yml.txt` runs it
after the daily drift check and shares the weekly retrain's concurrency group.

## Prune old models

```
python -m pm25_forecast.registry_cli prune --keep 4            # dry run: prints keep/delete lines
python -m pm25_forecast.registry_cli prune --keep 4 --confirm  # deletes
```

Always kept: the production champion, `CHALLENGER_MODEL_VERSION`, the rollback target (the
`previous_version` of the latest logged change), every model with predictions
issued in the last 30 days, and the `--keep` newest models by registration time. The rest are
deleted from `models` (and their cached drift reference); drift check history and `model_events`
stay. The delete statement re-checks the champion and recent predictions, so a referenced model is
never deleted. Deleted models cannot be restored (re-register from the original bundle).

## Weekly housekeeping

`python -m pm25_forecast.housekeeping` (Sundays 04:17 UTC, `housekeeping-workflow.yml.txt`) counts
and then deletes model prediction rows older than 90 days (`--dry-run` only counts), and lists what
`prune --keep 4` would delete. It never deletes a model; the owner reviews the list in the
`RUN_SUMMARY` and runs `prune --confirm` by hand. The purge only touches model prediction rows:
CAMS vintages (`model_version = 'cams_global'`) are never purged, because training and the drift
reference use them as the `cams_pm25` input.

## Exit codes

0 ok or skipped because training data is not ready. 1 retraining blocked because the holdout
boundary is unverified, or failed (DB down, unscorable champion, other training errors, crash);
the condition is alerted. The retrain CLI's specific alert suppresses the workflow's generic
failure alert when it completes; the generic alert remains the fallback if notification fails
before the CLI can mark its output. External webhook delivery cannot be guaranteed.
2 configuration error (no `DATABASE_URL`; `AUTO_PROMOTE` other than true, false or blank).
