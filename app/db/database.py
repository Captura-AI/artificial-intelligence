import psycopg2
import psycopg2.extras


def get_connection(database_url: str) -> psycopg2.extensions.connection:
    """Open a PostgreSQL connection using the DATABASE_URL connection string."""
    return psycopg2.connect(database_url)


def init_db(database_url: str) -> None:
    """Create the plate_results table if it does not already exist."""
    conn = get_connection(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_results (
                    id          SERIAL PRIMARY KEY,
                    moment_id   TEXT        NOT NULL,
                    file_path   TEXT,
                    plate_text  TEXT,
                    confidence  DOUBLE PRECISION,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        conn.commit()
    finally:
        conn.close()
