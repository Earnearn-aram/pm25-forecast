"""KAN-80: frozen, hashable experiment dataset exported read-only from the hosted database.

`python -m pm25_forecast.experiments.dataset export --out <dir>` writes two Parquet files and a
`manifest.json` to a directory outside git. `load_snapshot(dir)` reads them back as
`backtest.Observation` / `backtest.Vintage` objects after re-verifying the content hash.

Availability: an air4thai observation's `available_at` is `ts + 1 h` (KAN-79 review: backfilled
rows' `ingested_at` is days after `ts`); the real `ingested_at` is kept as a separate audit
column. Other sources use `ingested_at`. A CAMS vintage's `available_at` is `created_at`; stored
`cams_global` maps to model `cams`. Null observation values are omitted, never filled. The
database is only read, inside a `SET TRANSACTION READ ONLY` transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from pm25_forecast.backtest import Observation, Vintage
from pm25_forecast.collector import ConfigError
from pm25_forecast.drift import DATA_SOURCE_BREAK
from pm25_forecast.settings import get_settings
from pm25_forecast.storage import db

logger = logging.getLogger(__name__)

# KAN-74: targets from this instant were affected by the 24 h rolling-average headline value;
# they are excluded unless the owner confirms the repair (`--kan74-repaired`). A target exactly
# at the cut-off is excluded (conservative).
KAN74_CUTOFF = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)
SCHEMA_VERSION = 1
HOUR = timedelta(hours=1)
AIR4THAI_LAG = HOUR
DEFAULT_SNAPSHOT_ROOT = Path("~/pm25-experiments/snapshots")
DB_VINTAGE_MODEL = "cams_global"
VINTAGE_MODEL = "cams"
OBSERVATIONS_FILE = "observations.parquet"
VINTAGES_FILE = "vintages.parquet"
MANIFEST_FILE = "manifest.json"

OBSERVATIONS_QUERY = """
SELECT station_id, ts, pm25, source, ingested_at
FROM observations
WHERE pm25 IS NOT NULL
ORDER BY station_id, ts
"""
VINTAGES_QUERY = """
SELECT station_id, issue_time, valid_time, pm25_pred, created_at AS available_at, model_version
FROM predictions
WHERE model_version = 'cams_global'
ORDER BY station_id, issue_time, valid_time
"""

# Row layouts: observations (station, ts, pm25, source, available_at, ingested_at);
# vintages (station, issue_time, valid_time, value, available_at, model).
ObsRow = tuple[str, datetime, float, str, datetime, datetime]
VinRow = tuple[str, datetime, datetime, float, datetime, str]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value) and value >= 0


def fetch_rows(
    database_url: str, *, connect: Callable[[str], Any] = db.connect
) -> tuple[list[tuple], list[tuple]]:
    """Read observations and CAMS vintages in one read-only transaction."""
    with connect(database_url) as source:
        # PostgreSQL rejects every data-modifying statement within this transaction. It must be
        # the first statement of the transaction.
        source.execute("SET TRANSACTION READ ONLY")
        obs = source.execute(OBSERVATIONS_QUERY).fetchall()
        vin = source.execute(VINTAGES_QUERY).fetchall()
    return list(obs), list(vin)


def normalise_and_filter(
    obs_rows: Iterable[tuple],
    vin_rows: Iterable[tuple],
    *,
    kan74_repaired: bool = False,
) -> tuple[list[ObsRow], list[VinRow]]:
    """Apply the break/cut-off filters, drop invalid values, return sorted UTC rows."""

    def keep(target: datetime) -> bool:
        return target >= DATA_SOURCE_BREAK and (kan74_repaired or target < KAN74_CUTOFF)

    obs: list[ObsRow] = []
    for station, ts, pm25, source, ingested_at in obs_rows:
        ts, ingested_at = _utc(ts), _utc(ingested_at)
        if _finite(pm25) and keep(ts):
            # Walk-forward rule: an hourly air4thai reading is treated as available one hour
            # after its timestamp; backfilled rows' ingested_at (days later) is kept for audit.
            available_at = ts + AIR4THAI_LAG if source == "air4thai" else ingested_at
            obs.append((str(station), ts, float(pm25), str(source), available_at, ingested_at))
    vin: list[VinRow] = []
    for station, issue, valid, value, available_at, model in vin_rows:
        issue, valid, available_at = _utc(issue), _utc(valid), _utc(available_at)
        model = VINTAGE_MODEL if model == DB_VINTAGE_MODEL else str(model)
        if _finite(value) and keep(valid):
            vin.append((str(station), issue, valid, float(value), available_at, model))
    obs.sort()
    vin.sort()
    return obs, vin


def content_sha256(obs: Sequence[ObsRow], vin: Sequence[VinRow]) -> str:
    """SHA-256 of a canonical serialisation: sorted rows, UTC ISO times, repr floats."""
    digest = hashlib.sha256()

    def feed(tag: str, rows: Iterable[tuple]) -> None:
        digest.update(f"#{tag}\n".encode())
        for row in rows:
            cells = [_iso(c) if isinstance(c, datetime) else c for c in row]
            digest.update(json.dumps(cells, separators=(",", ":"), ensure_ascii=True).encode())
            digest.update(b"\n")

    feed("observations", sorted(obs))
    feed("vintages", sorted(vin))
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_sha() -> str:
    try:
        return _git("rev-parse", "HEAD") + ("-dirty" if _git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(
    obs: Sequence[ObsRow],
    vin: Sequence[VinRow],
    *,
    kan74_repaired: bool,
    created_at: datetime | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    created_at = _utc(created_at or datetime.now(UTC))
    stations = sorted({r[0] for r in obs} | {r[0] for r in vin})
    obs_hours: dict[str, set[datetime]] = defaultdict(set)
    obs_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for station, ts, _, source, *_ in obs:
        obs_hours[station].add(ts)
        obs_counts[station][source] += 1
    vin_counts: dict[str, int] = defaultdict(int)
    for station, *_ in vin:
        vin_counts[station] += 1

    first = min((r[1] for r in obs), default=None)
    last = max((r[1] for r in obs), default=None)
    expected = int((last - first) / HOUR) + 1 if first and last else 0
    missing = {
        s: round(1 - len(obs_hours[s]) / expected, 6) if expected else None for s in stations
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _iso(created_at),
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "content_sha256": content_sha256(obs, vin),
        "files": {"observations": OBSERVATIONS_FILE, "vintages": VINTAGES_FILE},
        "time_range": {
            "first_usable_target": _iso(first) if first else None,
            "last_usable_target": _iso(last) if last else None,
            "first_vintage_valid_time": _iso(min(r[2] for r in vin)) if vin else None,
            "last_vintage_valid_time": _iso(max(r[2] for r in vin)) if vin else None,
            "first_vintage_issue_time": _iso(min(r[1] for r in vin)) if vin else None,
            "last_vintage_issue_time": _iso(max(r[1] for r in vin)) if vin else None,
        },
        "stations": stations,
        "row_counts": {
            "observations_total": len(obs),
            "vintages_total": len(vin),
            "observations_by_station_source": {
                s: dict(sorted(c.items())) for s, c in sorted(obs_counts.items())
            },
            "vintages_by_station": dict(sorted(vin_counts.items())),
        },
        "missing_hour_rate_by_station": missing,
        "missing_hour_rate_note": (
            "1 - observed hours / hours in [first_usable_target, last_usable_target]; "
            "usable = non-null finite pm25"
        ),
        "filters": {
            "data_source_break_utc": _iso(DATA_SOURCE_BREAK),
            "kan74_cutoff_utc": _iso(KAN74_CUTOFF),
            "kan74_repaired": kan74_repaired,
            "rule": (
                "keep observation ts / vintage valid_time >= data_source_break_utc"
                + ("" if kan74_repaired else " and < kan74_cutoff_utc")
                + "; null, negative and non-finite values dropped; cams_global -> model cams"
            ),
            "available_at": (
                "observations: air4thai available_at = ts + 1 h (backfilled ingested_at would "
                "distort walk-forward training; real ingested_at kept as a separate column for "
                "audit); other sources: ingested_at. vintages: created_at"
            ),
        },
    }


def _write_parquet(path: Path, columns: str, rows: Sequence[tuple]) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"CREATE TABLE t ({columns})")
        if rows:
            marks = ", ".join("?" for _ in rows[0])
            con.executemany(f"INSERT INTO t VALUES ({marks})", rows)
        con.execute(f"COPY (SELECT * FROM t) TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


def write_snapshot(
    out: Path,
    obs: Sequence[ObsRow],
    vin: Sequence[VinRow],
    *,
    kan74_repaired: bool,
    created_at: datetime | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"snapshot directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        obs, vin, kan74_repaired=kan74_repaired, created_at=created_at, git_sha=git_sha
    )
    _write_parquet(
        out / OBSERVATIONS_FILE,
        "station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, source VARCHAR, "
        "available_at TIMESTAMPTZ, ingested_at TIMESTAMPTZ",
        obs,
    )
    _write_parquet(
        out / VINTAGES_FILE,
        "station_id VARCHAR, issue_time TIMESTAMPTZ, valid_time TIMESTAMPTZ, value DOUBLE, "
        "available_at TIMESTAMPTZ, model VARCHAR",
        vin,
    )
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _read_parquet(path: Path) -> list[tuple]:
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        return con.execute(f"SELECT * FROM read_parquet('{path}')").fetchall()
    finally:
        con.close()


def load_snapshot(path: Path | str) -> tuple[list[Observation], list[Vintage], dict[str, Any]]:
    """Load a snapshot directory, verifying its content hash against the manifest."""
    path = Path(path).expanduser()
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    obs_rows = [
        (s, _utc(ts), float(v), src, _utc(av), _utc(ing))
        for s, ts, v, src, av, ing in _read_parquet(path / OBSERVATIONS_FILE)
    ]
    vin_rows = [
        (s, _utc(i), _utc(v), float(val), _utc(av), m)
        for s, i, v, val, av, m in _read_parquet(path / VINTAGES_FILE)
    ]
    if content_sha256(obs_rows, vin_rows) != manifest["content_sha256"]:
        raise ValueError(f"snapshot content hash mismatch in {path}")
    observations = [Observation(s, ts, v, av) for s, ts, v, _src, av, _ing in sorted(obs_rows)]
    vintages = [Vintage(s, i, v, val, av, m) for s, i, v, val, av, m in sorted(vin_rows)]
    return observations, vintages, manifest


def export_snapshot(
    out: Path,
    database_url: str,
    *,
    kan74_repaired: bool = False,
    connect: Callable[[str], Any] = db.connect,
) -> dict[str, Any]:
    raw_obs, raw_vin = fetch_rows(database_url, connect=connect)
    obs, vin = normalise_and_filter(raw_obs, raw_vin, kan74_repaired=kan74_repaired)
    return write_snapshot(out, obs, vin, kan74_repaired=kan74_repaired)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_SNAPSHOT_ROOT / f"snapshot-{stamp}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export a frozen snapshot (read-only DB access)")
    export.add_argument("--out", type=Path, default=None, help="empty/new directory outside git")
    export.add_argument(
        "--kan74-repaired",
        action="store_true",
        help="keep targets at/after the KAN-74 cut-off (only once the repair is confirmed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args = _parse_args(argv)
        out = (args.out or _default_out()).expanduser().resolve()
        try:
            root = Path(_git("rev-parse", "--show-toplevel")).resolve()
        except (OSError, subprocess.CalledProcessError):
            root = None
        if root is not None and (out == root or root in out.parents):
            raise ConfigError("--out must be outside the git repository")
        settings = get_settings()
        if settings.database_url is None:
            raise ConfigError("DATABASE_URL is not set")
        manifest = export_snapshot(
            out, settings.database_url.get_secret_value(), kan74_repaired=args.kan74_repaired
        )
        logger.info("snapshot written to %s", out)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (ConfigError, FileExistsError) as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception as exc:
        # Driver exceptions can include connection details; emit only the exception type.
        logger.error("dataset export failed (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
