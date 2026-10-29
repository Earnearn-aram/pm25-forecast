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

## The rule

Promote only if, at +24 h, the candidate has at least 100 matched pairs, its MAE is at least 2%
lower (relative) than the champion's, and it beats persistence. With no champion, only the pairs
and persistence checks apply. Anything else is keep. A champion that cannot be loaded or scored
is never replaced (the run fails and alerts).

Production changes only if the repo variable `AUTO_PROMOTE` is `true` (default: unset = off).
Otherwise a promote decision only alerts `recommend promote <version>`.

## Promote manually

```
python -m pm25_forecast.registry_cli check retrain-20261102
python -m pm25_forecast.registry_cli promote retrain-20261102 --confirm
```

`DATABASE_URL` comes from the environment or `.env`. `champion` shows the current pointer.

## Roll back

Registered versions are immutable, so a rollback is a promote of the previous champion. The
previous version is printed by `promote` (`production champion: A -> B`) and logged in the
`RUN_SUMMARY` (`roles.champion`) of the run that replaced it:

```
python -m pm25_forecast.registry_cli promote <previous-version> --confirm
```

Production then points at it again. Also unset `AUTO_PROMOTE` if auto-promotion caused it.

## Exit codes

0 ok. 1 failed (DB down, not enough data, unscorable champion, crash); the failure is alerted.
2 configuration error (no `DATABASE_URL`, unparseable `AUTO_PROMOTE`).
