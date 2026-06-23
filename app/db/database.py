import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

# One connection pool per database URL, created lazily and reused for the process
# lifetime. Pooling avoids the open/connect/close-per-call overhead that made the
# service slow and fragile under the concurrent load of bulk uploads.
_POOLS: dict[str, pg_pool.ThreadedConnectionPool] = {}
_POOLS_LOCK = threading.Lock()


def _get_pool(
    database_url: str,
    min_size: int = 1,
    max_size: int = 10,
) -> pg_pool.ThreadedConnectionPool:
    """Return the pool for ``database_url``, creating it on first use.

    Double-checked locking keeps creation thread-safe — FastAPI serves sync work
    from a thread pool and /analyze runs in worker threads, so multiple threads
    may race to create the pool. The first caller's sizing wins.
    """
    existing = _POOLS.get(database_url)
    if existing is not None:
        return existing

    with _POOLS_LOCK:
        existing = _POOLS.get(database_url)
        if existing is None:
            existing = pg_pool.ThreadedConnectionPool(min_size, max_size, dsn=database_url)
            _POOLS[database_url] = existing
        return existing


@contextmanager
def get_connection(database_url: str) -> Iterator[psycopg2.extensions.connection]:
    """Borrow a pooled connection, returning it to the pool on exit.

    The caller is responsible for commit/rollback; the connection is always
    returned to the pool (not closed) so it can be reused.
    """
    pool = _get_pool(database_url)
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        # Clear the aborted transaction so the next borrower gets a clean session.
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db(database_url: str, min_size: int = 1, max_size: int = 10) -> None:
    """Create the pool, then create/upgrade the plate_results table."""
    _get_pool(database_url, min_size=min_size, max_size=max_size)

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_results (
                    id                     SERIAL PRIMARY KEY,
                    moment_id              TEXT        NOT NULL,
                    file_path              TEXT,
                    plate_text             TEXT,
                    confidence             DOUBLE PRECISION,
                    motor_type             TEXT,
                    motor_type_confidence  DOUBLE PRECISION,
                    color                  TEXT,
                    color_confidence       DOUBLE PRECISION,
                    created_at             TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            # In-place migration for databases created before the
            # motor type / color columns existed.
            cur.execute(
                """
                ALTER TABLE plate_results
                    ADD COLUMN IF NOT EXISTS motor_type            TEXT,
                    ADD COLUMN IF NOT EXISTS motor_type_confidence DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS color                 TEXT,
                    ADD COLUMN IF NOT EXISTS color_confidence      DOUBLE PRECISION
                """
            )
        conn.commit()
