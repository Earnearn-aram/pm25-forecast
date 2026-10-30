# Weekly retrain (KAN-23)

`python -m pm25_forecast.retrain` runs every Monday 03:17 UTC (see `retrain-workflow.yml.txt`) and
on demand.

## What runs

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

Production changes only if the repo variable `AUTO_PROMOTE` is `true` (default: unset = off).
Otherwise a promote decision only alerts `recommend promote <version>`.

## Promote manually

```
python -m pm25_forecast.registry_cli check retrain-20261102
python -m pm25_forecast.registry_cli promote retrain-20261102 --confirm
```

`DATABASE_URL` comes from the environment or `.env`. `champion` shows the current pointer.

## The promotion log (`model_events`)

Every champion change and every retrain decision is one row in `model_events` (migration
`005_model_events.sql`): `at` (UTC), `action` (`promote`, `rollback`, `recommend`, `keep`),
`model_version`, `previous_version`, `actor` (`cli`, `retrain` or `auto`), `metrics` (JSON, the
comparison numbers) and `reason`. A `promote` or `rollback` row is written in the same transaction
as the champion pointer change, so the log and the pointer cannot disagree. `retrain` writes a
`promote` row (actor `auto`) when `AUTO_PROMOTE=true`, otherwise `recommend` (and alerts), and a
`keep` row when the candidate is not good enough.

```sql
SELECT at, action, model_version, previous_version, actor, reason
FROM model_events ORDER BY at DESC LIMIT 20;
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
older one. Also unset `AUTO_PROMOTE` if auto-promotion caused the problem.

## Drift-triggered retrain

`python -m pm25_forecast.retrain --if-drift` exits 0 without training unless the latest drift run
(`drift.latest_drift_alert`, at most 36 h old) has an alert for `CHALLENGER_MODEL_VERSION`, or for
the production champion when no challenger is set. With no challenger and no champion it does
nothing. When it does run it is the normal retrain: the candidate is registered and, unless
`AUTO_PROMOTE=true`, only recommended. The `drift-retrain` job in `drift-workflow.yml.txt` runs it
after the daily drift check and shares the weekly retrain's concurrency group.

## Prune old models

```
python -m pm25_forecast.registry_cli prune --keep 4            # dry run: prints keep/delete lines
python -m pm25_forecast.registry_cli prune --keep 4 --confirm  # deletes
```

Always kept: the production champion, `CHALLENGER_MODEL_VERSION`, every model with predictions
issued in the last 30 days, and the `--keep` newest models by registration time. The rest are
deleted from `models` (and their cached drift reference); drift check history and `model_events`
stay. The delete statement re-checks the champion and recent predictions, so a referenced model is
never deleted. Deleted models cannot be restored (re-register from the original bundle).

## Weekly housekeeping

`python -m pm25_forecast.housekeeping` (Sundays 04:17 UTC, `housekeeping-workflow.yml.txt`) counts
and then deletes prediction rows older than 90 days (`--dry-run` only counts), and lists what
`prune --keep 4` would delete. It never deletes a model; the owner reviews the list in the
`RUN_SUMMARY` and runs `prune --confirm` by hand. The 90-day purge also removes old CAMS vintages;
a drift reference is cached per model, so only a rebuild after a training-window change needs
data that old.

## Exit codes

0 ok. 1 failed (DB down, not enough data, unscorable champion, crash); the failure is alerted.
2 configuration error (no `DATABASE_URL`; `AUTO_PROMOTE` other than true, false or blank).
