"""Run logs (KAN-19): UTC timestamps on every line and one JSON summary line per run."""

import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
SUMMARY_TAG = "RUN_SUMMARY"


class UtcFormatter(logging.Formatter):
    """ISO 8601 UTC timestamps (``2026-10-27T01:02:03.456Z``), whatever the host timezone."""

    converter = time.gmtime


def configure_logging(level: int = logging.INFO) -> None:
    """Root handler with UTC timestamps (like basicConfig: a no-op if the root logger already
    has handlers, e.g. under pytest). httpx is quietened: its INFO lines log full URLs."""
    handler = logging.StreamHandler()
    handler.setFormatter(UtcFormatter(LOG_FORMAT, DATE_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def format_summary(job: str, fields: Mapping[str, Any]) -> str:
    """``RUN_SUMMARY {"job": ..., ...}``: one grep-able, machine-readable line."""
    return f"{SUMMARY_TAG} " + json.dumps({"job": job, **fields}, sort_keys=True, default=_jsonable)


def log_summary(logger: logging.Logger, job: str, fields: Mapping[str, Any]) -> None:
    logger.info("%s", format_summary(job, fields))
