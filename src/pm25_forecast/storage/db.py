"""PostgreSQL connection helper (psycopg 3) and the driver's connection errors for the guard."""

import logging

import psycopg
from tenacity import Retrying, stop_after_attempt, wait_exponential
from tenacity.retry import retry_if_exception

logger = logging.getLogger(__name__)

# Passed to `run_entrypoint(connection_errors=...)`: these become DatabaseUnavailableError.
# OperationalError covers refused/timed-out/dropped connections and a server that is shutting
# down; InterfaceError covers use of a connection that is already closed.
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
)

CONNECT_TIMEOUT_S = 15  # Neon wakes from idle in about a second; leave room for retries
CONNECT_ATTEMPTS = 3
RETRY_WAIT = wait_exponential(multiplier=1, max=4)  # module level so tests can replace it


def _is_transient(exc: BaseException) -> bool:
    """OperationalError worth retrying: not a permanent auth (class 28) or unknown-DB error."""
    if not isinstance(exc, psycopg.OperationalError):
        return False
    state = getattr(exc, "sqlstate", None) or ""
    return not (state.startswith("28") or state == "3D000")


def _log_retry(state) -> None:
    # Type only: connection error messages name hosts, and the URL is a secret.
    exc = state.outcome.exception()
    logger.warning(
        "database connect attempt %d failed (%s); retrying",
        state.attempt_number,
        type(exc).__name__,
    )


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection (use as a context manager: commit on success, rollback on error).

    Retries transient `OperationalError` (a cold Neon compute can refuse the first attempt) up to
    CONNECT_ATTEMPTS times, but not permanent auth/unknown-database errors (SQLSTATE 28xxx and
    3D000). Only the connect is retried, never work done on the connection.
    The URL is only ever passed to the driver; it is never logged or put in an exception here.
    """
    for attempt in Retrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(CONNECT_ATTEMPTS),
        wait=RETRY_WAIT,
        before_sleep=_log_retry,
        reraise=True,
    ):
        with attempt:
            return psycopg.connect(database_url, connect_timeout=CONNECT_TIMEOUT_S)
    raise AssertionError("unreachable")  # pragma: no cover
