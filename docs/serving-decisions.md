# Champion serving decisions (KAN-119)

The first transaction in an issue hour that produces a served forecast freezes the current
champion pointer in `serving_issue_decisions`. Selection occurs during the run, not at the
floored issue boundary; `selected_at` is the database clock at selection and is not a commit
timestamp. Once a decision is frozen, a promotion or rollback becomes eligible at the next issue
hour. If it occurs before the first successful selection (including after an empty run), the new
pointer can be selected in the current hour. Repeated or concurrent runs reuse a committed decision.

For each station and horizon with a usable forecast, `serving_forecasts` records the immutable
first serving choice and the `run_id` of the matching row in `predictions`. The chosen row may
be an existing vintage from an earlier run; the run ID identifies it rather than claiming it was
newly generated. When the chosen champion is unavailable, unsupported for a horizon or lacks
eligible inputs, persistence is recorded as `baseline_fallback`; if no eligible forecast exists,
there is no serving record and a later run may recover it. Existing forecasts are not assigned
roles retroactively. Models produced in shadow or later retries can exist in `predictions` without
being the serving choice. Missing/corrupt models produce an alert, not fabricated predictions.

## Deployment and rollback

The owner must apply `src/pm25_forecast/storage/migrations/006_serving_decisions.sql` to the
target database **before deploying this code**. Verify that both new tables exist and confirm a
backup and rollback plan first. If migration is unavailable, do not deploy: Predict requires the
tables and would fail before producing the baseline. Apply the migration once; it is idempotent.
After deployment, verify a fresh issue hour and query the decision and per-station serving rows,
including the matching prediction's `run_id`. Do not infer historical roles from current pointer.

Rollback the code to its previous version if verification fails. Retain the new tables and their
records for audit; do not drop them while the new code may run. Rollback of a champion pointer is
effective at the next selection, not for an already frozen decision. `selected_at` and
`issue_time` must not be interpreted as proof that a forecast was available at the hour boundary.

Dashboard and Score still read the configured shadow challenger(s) (a list since KAN-104); they do not consume this serving
ledger. KAN-120/121 must integrate those consumers before presenting the ledger as end-to-end
production serving or model performance. Auto-promotion remains outside this change.

Serving rows reference the persisted prediction and are deleted when housekeeping purges that
prediction. Hourly selection records are retained for audit; their model version is historical
text and does not prevent registry pruning. For a retained forecast, join on station, issue time,
valid time and model version; do not treat an old selection without a forecast as a live forecast.
