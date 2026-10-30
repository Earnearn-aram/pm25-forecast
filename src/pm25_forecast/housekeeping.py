"""Weekly housekeeping (KAN-24). Run with ``python -m pm25_forecast.housekeeping``.

1. Predictions retention: count, then delete, the prediction rows older than 90 days
   (``sql.PURGE_OLD_PREDICTIONS``). ``--dry-run`` only counts.
2. Registry prune in DRY-RUN mode only: logs which ``models`` rows ``registry_cli prune`` would
   delete. This job never deletes a model; the owner decides (see docs/retrain.md).

One RUN_SUMMARY line holds both results. DB down: the run is skipped, alerted, exit 1.
Configuration error: exit 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from pm25_forecast.storage.registry import plan_prune

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2
PRUNE_KEEP = 4  # newest models the dry-run prune would keep (as the registry_cli default)


@dataclass
class HousekeepingResult:
    run_id: str
    status: str = "ok"
    dry_run: bool = False
    predictions_due: int = 0
    predictions_deleted: int = 0
    prune_keep: dict[str, str] = field(default_factory=dict)
    prune_would_delete: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    dry_run: bool = False,
) -> HousekeepingResult:
    """One housekeeping run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = clock()
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    result = HousekeepingResult(run_id=default_run_id(now), dry_run=dry_run)
    url = settings.database_url.get_secret_value()

    with open_db(connect, url) as con:
        result.predictions_due = int(con.execute(sql.COUNT_OLD_PREDICTIONS, {}).fetchone()[0])
        logger.info("predictions older than 90 days: %d", result.predictions_due)
        if not dry_run:
            cursor = con.execute(sql.PURGE_OLD_PREDICTIONS, {})
            result.predictions_deleted = int(cursor.rowcount)
            logger.info("deleted %d old prediction rows", result.predictions_deleted)
        plan = plan_prune(con, PRUNE_KEEP, challenger=settings.challenger_model_version, now=now)
        result.prune_keep = plan.keep
        result.prune_would_delete = plan.delete  # dry run only: nothing is deleted here
    for version in plan.delete:
        logger.info("registry prune (dry run) would delete %s", version)
    log_summary(
        logger,
        "housekeeping",
        {
            "run_id": result.run_id,
            "status": result.status,
            "dry_run": result.dry_run,
            "predictions_due": result.predictions_due,
            "predictions_deleted": result.predictions_deleted,
            "prune_mode": "dry_run",
            "prune_keep": result.prune_keep,
            "prune_would_delete": result.prune_would_delete,
            "failures": result.failures,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 housekeeping]")
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--dry-run", action="store_true", help="count old predictions but delete nothing"
        )
        args = parser.parse_args(argv)
        try:
            settings = get_settings()
        except ValidationError as exc:  # report field names only, never values
            fields = ", ".join(sorted({".".join(map(str, e["loc"])) for e in exc.errors()}))
            raise ConfigError(f"invalid settings: {fields}") from None
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 housekeeping]")
        result, code = run_entrypoint(
            lambda: run(settings, dry_run=args.dry_run),
            alert,
            connection_errors=db.CONNECTION_ERRORS,
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "housekeeping", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.error("housekeeping crashed (%s)", type(exc).__name__)
        safe_alert(alert, f"housekeeping crashed: {type(exc).__name__}")
        log_summary(logger, "housekeeping", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "housekeeping", {"status": "db_unreachable"})
        return code
    return code


if __name__ == "__main__":
    sys.exit(main())
