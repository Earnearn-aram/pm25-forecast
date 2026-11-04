# Training data readiness (KAN-118)

The readiness CLI currently fails closed. The repository has no trusted, versioned provenance
source that binds an operational export to a non-holdout boundary. A timestamp supplied by an
operator cannot prove that boundary. Therefore the CLI validates the requested cutoff, reports
`blocked_unverified_export_boundary`, and returns exit code 1. It has no export-path argument and
does not open data or count rows. Do not treat this blocker report as evidence that operational
training data is ready.

```bash
uv run python -m pm25_forecast.training_readiness \
  --cutoff '<UTC_CUTOFF>'
```

The command accepts only a cutoff. Its JSON output confirms `export_opened: false`. When a trusted
boundary source exists, a future report may expose usable examples, trainer/validation counts, and
a `count_ready` status. Those counts would still not prove model quality or promotion eligibility.

The live observations table is mutable. If a value is corrected, its earlier value and arrival
time are not retained; current `ingested_at` is not proof of what was known earlier, nor of exact
transaction commit visibility. A late backfill is not made available retroactively. See KAN-115
for the capture/replay design and its migration and retention decisions. Do not relax the
availability cutoff or invent history to make the report ready.

KAN-115 must define immutable captures, trusted boundary provenance, replay behavior, retention,
and a controlled rollout before operational readiness reporting can resume. Existing training and
alert behavior is unchanged. This ticket does not authorize retraining or promotion.
