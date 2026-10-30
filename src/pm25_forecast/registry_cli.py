"""Owner commands for the model registry (KAN-29): register a local bundle, check, manually promote.

    python -m pm25_forecast.registry_cli register --artifact model.json --manifest manifest.json
    python -m pm25_forecast.registry_cli check VERSION
    python -m pm25_forecast.registry_cli champion
    python -m pm25_forecast.registry_cli promote VERSION --confirm
    python -m pm25_forecast.registry_cli rollback --confirm
    python -m pm25_forecast.registry_cli prune [--keep 4] [--confirm]

``promote`` and ``rollback`` write one ``model_events`` row in the same transaction as the pointer
change (KAN-24). ``rollback`` moves the pointer back to the previous champion recorded there and
refuses if none is recorded or that model cannot be served. ``prune`` deletes old ``models`` rows
(dry run unless ``--confirm``), always keeping the champion, ``CHALLENGER_MODEL_VERSION``, models
with predictions in the last 30 days, and the newest ``--keep``.

``register`` publishes an immutable version and reads it back (KAN-16); it never moves the
champion pointer. ``check`` loads a registered version exactly as the hourly predict step would
(bundle shape and feature spec), so a version can be verified before it is configured as
``CHALLENGER_MODEL_VERSION``. ``promote`` is the manual champion flip; it refuses to run without
``--confirm`` and refuses a version the serving loader rejects. ``DATABASE_URL`` comes from the
environment or ``.env`` and is never printed.

Exit codes: 0 ok, 1 failed, 2 configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pm25_forecast.collector import open_db
from pm25_forecast.models.serving import ModelLoadError, load_registered_model
from pm25_forecast.settings import get_settings
from pm25_forecast.storage import db, registry

EXIT_FAILED, EXIT_CONFIG = 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm25_forecast.registry_cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="publish a local bundle and manifest (no promotion)")
    reg.add_argument("--artifact", type=Path, required=True)
    reg.add_argument("--manifest", type=Path, required=True)
    chk = sub.add_parser("check", help="load a registered version as the predict step would")
    chk.add_argument("version")
    sub.add_parser("champion", help="show the current production champion")
    pro = sub.add_parser("promote", help="manually point production at a registered version")
    pro.add_argument("version")
    pro.add_argument("--confirm", action="store_true", help="required: this changes production")
    rol = sub.add_parser("rollback", help="point production back at the previous champion")
    rol.add_argument("--confirm", action="store_true", help="required: this changes production")
    pru = sub.add_parser("prune", help="delete old model rows (dry run unless --confirm)")
    pru.add_argument("--keep", type=int, default=4, help="newest models to keep (default 4)")
    pru.add_argument("--confirm", action="store_true", help="actually delete (default: dry run)")
    return parser


def prune_report(plan: registry.PrunePlan) -> list[str]:
    lines = [f"keep {v}: {why}" for v, why in plan.keep.items()]
    lines += [f"delete {v}" for v in plan.delete]
    return lines


def run(
    argv: Sequence[str] | None = None,
    *,
    database_url: str | None = None,
    connect: Callable[[str], Any] = db.connect,
    out: Callable[[str], None] = print,
) -> int:
    args = _parser().parse_args(argv)
    if database_url is None:
        secret = get_settings().database_url
        if secret is None:
            out("configuration error: DATABASE_URL is not set")
            return EXIT_CONFIG
        database_url = secret.get_secret_value()

    try:
        if args.command == "register":
            artifact = args.artifact.read_bytes()
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            version = manifest.get("version", "")
            digest = registry.publish_model(
                database_url, version, artifact, manifest, connect=connect
            )
            out(f"registered {version} sha256={digest} (champion unchanged)")
            return 0
        with open_db(connect, database_url) as con:
            if args.command == "check":
                model = load_registered_model(con, args.version)
                out(
                    f"ok {model.model_version}: horizons={list(model.horizons)} "
                    f"features={list(model.feature_list)}"
                )
                return 0
            if args.command == "champion":
                try:
                    version, _artifact, _manifest = registry.read_champion(con)
                except LookupError:
                    out("no production champion is set")
                    return 0
                out(f"production champion: {version}")
                return 0
            if args.command == "prune":
                plan = registry.plan_prune(
                    con, args.keep, challenger=get_settings().challenger_model_version
                )
                for line in prune_report(plan):
                    out(line)
                if not args.confirm:
                    out(f"dry run: {len(plan.delete)} model(s) would be deleted; use --confirm")
                    return 0
                deleted = registry.prune_models(con, plan)
                out(f"deleted {len(deleted)} model(s)")
                return 0
            if args.command == "rollback":
                if not args.confirm:
                    out("refused: rollback changes production; re-run with --confirm")
                    return EXIT_FAILED
                target = registry.previous_champion(con)
                load_registered_model(con, target)  # never roll back to an unservable model
                current = registry.promote_model(
                    con, target, actor="cli", action="rollback", reason="manual rollback"
                )
                out(f"production champion: {current} -> {target} (rollback)")
                return 0
            # promote
            if not args.confirm:
                out("refused: promote changes production; re-run with --confirm")
                return EXIT_FAILED
            load_registered_model(con, args.version)  # never promote an unservable model
            previous = registry.promote_model(
                con, args.version, actor="cli", reason="manual promote"
            )
        out(f"production champion: {previous or 'none'} -> {args.version}")
        return 0
    except (ModelLoadError, LookupError, ValueError, OSError) as exc:
        out(f"failed: {exc}")
        return EXIT_FAILED


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
