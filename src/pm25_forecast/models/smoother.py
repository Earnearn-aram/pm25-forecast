"""One-parameter smoother challenger (KAN-76): ``latest + beta * (mean24 - latest)``.

``beta = 0`` is persistence, ``beta = 1`` is the trailing 24 h mean. The chosen ``beta`` and its
source (KAN-65 analysis) are recorded in the registered bundle and manifest. This module only
builds the registry artifact and manifest; the feature inputs come from ``pm25_forecast.features``
and the serving side is ``models/serving.py``.

Build the two files, then register them (owner step, see the KAN-76 PR)::

    uv run python -m pm25_forecast.models.smoother --out-dir out/kan76
    uv run python -m pm25_forecast.registry_cli register \
        --artifact out/kan76/artifact.json --manifest out/kan76/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pm25_forecast.features import SMOOTHER_FEATURE_NAMES

SMOOTHER_BETA = 0.5
SMOOTHER_HORIZONS: tuple[int, ...] = (24,)
SMOOTHER_VERSION = "kan76-smoother-24h-b05"
# KAN-65 v2 analysis (branch kan-65/ds-eda-features, out/kan65_v2/tables.md): KAN-15 snapshot and
# the +24 h HOLDOUT row (n = 10,729 station-hours) for the smoother and for persistence.
KAN65_SNAPSHOT_SHA256 = (
    "fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09"  # pragma: allowlist secret
)
KAN65_WINDOW = {"start": "2024-10-28T00:00:00+00:00", "end": "2026-10-28T21:00:00+00:00"}
KAN65_HOLDOUT_24H = {
    "n": 10729,
    "mae_smoother_beta_0.5": 4.114,
    "mae_persistence": 4.498,
    "diff_vs_persistence_7day_block_ci95": [-0.465, -0.297],
}


def smoother_artifact(
    beta: float = SMOOTHER_BETA, horizons: Sequence[int] = SMOOTHER_HORIZONS
) -> bytes:
    payload = {"format": 1, "smoother": {"beta": beta, "horizons": list(horizons)}}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def smoother_manifest(artifact: bytes, version: str = SMOOTHER_VERSION) -> dict[str, Any]:
    return {
        "version": version,
        "model_type": "exponential-smoother",
        "metrics": {"24": KAN65_HOLDOUT_24H},
        "training_window": dict(KAN65_WINDOW),  # nothing is fitted; the window beta was judged on
        "feature_list": list(SMOOTHER_FEATURE_NAMES),
        "data_sha256": KAN65_SNAPSHOT_SHA256,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "assumed_reporting_lag_hours": 1,
        "beta_source": "KAN-65 v2 analysis: fixed 0.5 (pre-declared, no same-regime validation)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = smoother_artifact()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "artifact.json").write_bytes(artifact)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(smoother_manifest(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"wrote {SMOOTHER_VERSION} to {args.out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
