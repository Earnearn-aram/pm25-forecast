# KAN-129: synthetic holdout-boundary proof

This first increment tests a metadata contract with synthetic inputs. It does not authorize an
operational export, retrain, promotion, or scoring of the experiment holdout. The KAN-126
fail-closed guard stays in force.

An independently reviewed anchor pins the boundary manifest digest, the frozen snapshot digest,
the snapshot-manifest digest, the protocol digest, the exact fold-derivation code digest, and a
protected lock reference. The boundary manifest records the same identities, the first usable
origin, the snapshot's last usable target, and the half-open holdout interval `[H0, E)`. The
verifier checks all pinned metadata and recomputes H0 and E from the protocol's fold code. A digest
inside the boundary manifest cannot establish its own trust: the anchor must be pinned in a
reviewed, protected commit or signed release. This increment does not create or approve an actual
production anchor.

For the fixture-only exclusion contract, observation `ts`, and forecast vintage `issue_time` and
`valid_time`, are each checked against `[H0, E)`. A vintage is excluded if either timestamp falls
inside that interval. Tests cover synthetic rows immediately before, at the start, inside, and at
the exclusive end. This conservative rule can exclude additional feature inputs; it does not
prove a production query applies the rule or that historical observations were available at their
forecast origin.

Before an operational path can use this contract, it needs a reviewed anchor for the actual frozen
snapshot, a source-bound export that filters before protected records are loaded, and a durable
export attestation binding the boundary digest, input/export hashes and exporter revision. It also
needs separately proven feature and label availability, source precedence, database transaction
visibility, origin semantics, cluster lineage, restore/failover handling, and sufficient mature
prospective data. Missing or inconsistent evidence must keep retraining blocked. No protected
experiment rows are read by these tests.

On 2026-11-05 the owner selected actual feature-read snapshot time as the canonical forecast
origin; the nominal UTC hour remains a grouping bucket. A later implementation must change
prediction, target/horizon alignment, maturity, scoring and dashboard presentation together and
prove snapshot-to-prediction lineage. This decision is not implemented by this proof and does not
make earlier observations or forecasts retroactively eligible.
