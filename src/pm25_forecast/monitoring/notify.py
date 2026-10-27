"""Job-failure alert for GitHub Actions (KAN-19). Run as an ``if: failure() || cancelled()`` step::

    python -m pm25_forecast.monitoring.notify failure   # env JOB_STATUS: ${{ job.status }}

It reads only the webhook (ALERT_WEBHOOK_URL) and GitHub's own run variables, never the
database or the collector's settings, so it still works when the job died before the collector
sent anything (crash, bad config, timeout, a failed install step).

Exit codes: 0 sent (or no webhook configured: warning logged), 1 delivery failed.
"""

import argparse
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime

from pm25_forecast.monitoring.alerts import post_message, warn_missing_webhook
from pm25_forecast.monitoring.logs import configure_logging

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_SEND_FAILED = 0, 1
_FIELD_MAX = 100  # GitHub variables are not secret, but keep the message short and bounded


def _field(env: Mapping[str, str], name: str, default: str = "?") -> str:
    value = " ".join(env.get(name, "").split())  # one line, no control characters
    return value[:_FIELD_MAX] or default


def run_url(env: Mapping[str, str]) -> str | None:
    """Link to the run (and attempt) from GitHub's default variables, or None outside Actions."""
    run_id = _field(env, "GITHUB_RUN_ID", "")
    repo = _field(env, "GITHUB_REPOSITORY", "")
    if not (run_id.isdigit() and repo):
        return None
    server = _field(env, "GITHUB_SERVER_URL", "https://github.com")
    url = f"{server}/{repo}/actions/runs/{run_id}"
    attempt = _field(env, "GITHUB_RUN_ATTEMPT", "")
    return f"{url}/attempts/{attempt}" if attempt.isdigit() else url


def failure_message(env: Mapping[str, str], now: datetime) -> str:
    """``[pm25 ci] FAILURE: workflow Collector, job collect, at <UTC>, run <url>``.

    The word after the prefix is JOB_STATUS (the workflow passes ``job.status``: ``failure``,
    or ``cancelled`` when the job hit its timeout or was cancelled), default ``failure``.
    """
    status = _field(env, "JOB_STATUS", "failure").upper()
    parts = [
        f"workflow {_field(env, 'GITHUB_WORKFLOW')}",
        f"job {_field(env, 'GITHUB_JOB')}",
        f"at {now.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"run {run_url(env) or 'n/a (not in GitHub Actions)'}",
    ]
    return f"[pm25 ci] {status}: " + ", ".join(parts)


def send(message: str, webhook_url: str | None) -> int:
    """Log `message` and post it. A missing webhook is a warning, not a crash."""
    logger.error("ALERT: %s", message)
    if not webhook_url:
        warn_missing_webhook("the job-failure alert")
        return EXIT_OK
    try:
        post_message(webhook_url, message)
    except Exception as exc:  # WebhookError messages never contain the URL
        logger.error("could not deliver the job-failure alert: %s", exc)
        return EXIT_SEND_FAILED
    logger.info("job-failure alert sent")
    return EXIT_OK


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m pm25_forecast.monitoring.notify")
    parser.add_argument("kind", choices=["failure"], help="what to report")
    parser.parse_args(argv)
    env = os.environ if env is None else env
    message = failure_message(env, now or datetime.now(UTC))
    return send(message, env.get("ALERT_WEBHOOK_URL", "").strip() or None)


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
