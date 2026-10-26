"""DB-down policy: skip the run and alert. Never fall back to stale or cached data."""

from collections.abc import Callable
from dataclasses import dataclass


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
        reason = f"predict run skipped: database unreachable ({type(exc).__name__})"
        alert(reason)
        return RunSkipped(reason)
