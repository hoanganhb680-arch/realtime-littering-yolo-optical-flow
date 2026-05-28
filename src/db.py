# ==============================================================================
# DB — SQLite storage cho lịch sử vi phạm
# ==============================================================================
import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

# Đặt DB ở thư mục gốc dự án (DA_TGMT/violations.db)
_ROOT  = Path(__file__).parent.parent.resolve()
DB_PATH = str(_ROOT / "violations.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS violations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id      INTEGER NOT NULL,
    trash_id       INTEGER NOT NULL,
    violation_type TEXT    NOT NULL,
    score          REAL    NOT NULL,
    timestamp      TEXT    NOT NULL,
    evidence_url   TEXT    NOT NULL,
    created_at     TEXT    DEFAULT (datetime('now','localtime'))
);
"""

def init_db() -> None:
    """Tạo bảng nếu chưa tồn tại."""
    with _conn() as conn:
        conn.execute(_CREATE_TABLE)

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------

def insert_violation(
    person_id: int,
    trash_id: int,
    violation_type: str,
    score: float,
    timestamp: str,
    evidence_url: str,
) -> int:
    sql = """
    INSERT INTO violations (person_id, trash_id, violation_type, score, timestamp, evidence_url)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    with _conn() as conn:
        cur = conn.execute(sql, (person_id, trash_id, violation_type, score, timestamp, evidence_url))
        return cur.lastrowid

def update_violation(
    row_id: int,
    trash_id: int,
    violation_type: str,
    score: float,
    timestamp: str,
    evidence_url: str,
) -> None:
    sql = """
    UPDATE violations
    SET trash_id = ?, violation_type = ?, score = ?, timestamp = ?, evidence_url = ?
    WHERE id = ?
    """
    with _conn() as conn:
        conn.execute(sql, (trash_id, violation_type, score, timestamp, evidence_url, row_id))

# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------

def get_all_violations(limit: int = 200, offset: int = 0) -> list[dict]:
    sql = """
    SELECT id, person_id, trash_id, violation_type, score, timestamp, evidence_url, created_at
    FROM violations
    ORDER BY id DESC
    LIMIT ? OFFSET ?
    """
    with _conn() as conn:
        rows = conn.execute(sql, (limit, offset)).fetchall()
    return [dict(r) for r in rows]

def count_violations() -> int:
    with _conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM violations").fetchone()
    return row["cnt"]

def clear_all_violations() -> None:
    """Xóa sạch bảng violations."""
    with _conn() as conn:
        conn.execute("DELETE FROM violations")
        # Reset AUTOINCREMENT
        conn.execute("DELETE FROM sqlite_sequence WHERE name='violations'")
