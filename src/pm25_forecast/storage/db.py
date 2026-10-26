"""PostgreSQL connection helper (psycopg 3) and the driver's connection errors for the guard."""

import psycopg

# Passed to `run_entrypoint(connection_errors=...)`: these become DatabaseUnavailableError.
# OperationalError covers refused/timed-out/dropped connections and a server that is shutting
# down; InterfaceError covers use of a connection that is already closed.
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
)

CONNECT_TIMEOUT_S = 15  # Neon wakes from idle in about a second; leave room for retries


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection (use as a context manager: commit on success, rollback on error).

    The URL is only ever passed to the driver; it is never logged or put in an exception here.
    """
    return psycopg.connect(database_url, connect_timeout=CONNECT_TIMEOUT_S)
