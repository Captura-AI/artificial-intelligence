from typing import Optional

from .database import get_connection


def save_plate_result(
    database_url: str,
    moment_id: str,
    file_path: Optional[str],
    plate_text: Optional[str],
    confidence: Optional[float],
) -> None:
    """Insert one plate-scan result row into plate_results."""
    conn = get_connection(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO plate_results (moment_id, file_path, plate_text, confidence)
                VALUES (%s, %s, %s, %s)
                """,
                (moment_id, file_path, plate_text, confidence),
            )
        conn.commit()
    finally:
        conn.close()
