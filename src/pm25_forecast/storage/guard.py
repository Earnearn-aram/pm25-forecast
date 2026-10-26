"""DB-down policy: skip the run and alert. Never fall back to stale or cached data."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (OSError, ConnectionError)


class DatabaseUnavailableError(Exception):
    """Raised by the storage layer when the database cannot be reached."""


@dataclass(frozen=True)
class RunSkipped:
    reason: str


def run_predict_step[T](step: Callable[[], T], alert: Callable[[str], None]) -> T | RunSkipped:
    """Run the predict step; if the DB is unreachable, alert and skip (no fallback)."""
    try:
        return step()
    except DatabaseUnavailableError as exc:
        # Cause type only (never str(exc): driver messages can name hosts or carry the URL).
        cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "unknown cause"
        reason = f"run skipped: database unreachable ({cause})"
        logger.error(reason)
        try:
            alert(reason)
        except Exception:
            logger.exception("alert failed while skipping run: %s", reason)
        return RunSkipped(reason)


def run_entrypoint[T](
    step: Callable[[], T],
    alert: Callable[[str], None],
    connection_errors: tuple[type[BaseException], ...] = DEFAULT_CONNECTION_ERRORS,
) -> tuple[T | RunSkipped, int]:
    """Entrypoint helper returning (result, exit_code).

    Driver-level connection errors (`connection_errors`, default OSError/ConnectionError; pass
    psycopg's error classes in KAN-18) become DatabaseUnavailableError. The OSError default is
    a placeholder: KAN-18 must set it to the driver's connection errors.
    The caller MUST `sys.exit(exit_code)`: a skipped run returns 1 so CI shows red, never green.
    Note: with the OSError default, non-DB OSErrors inside `step` are also treated as DB-down.
    """

    def guarded() -> T:
        try:
            return step()
        except DatabaseUnavailableError:
            raise
        except connection_errors as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    result = run_predict_step(guarded, alert)
    return result, 1 if isinstance(result, RunSkipped) else 0
