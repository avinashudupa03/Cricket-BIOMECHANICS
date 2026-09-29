"""SQLite persistence layer for pipeline and batch jobs.

Replaces the in-memory PIPELINE_JOBS / BATCH_JOBS dictionaries so job state
survives server restarts. The database file is stored at
``output_data/jobs.db`` and is created automatically on first use.

Usage:
    from database import db
    db.init_app(app)  # call once at startup

    # Pipeline jobs
    db.create_pipeline_job(job_id, video_path, video_name, shot_type, digest)
    db.update_pipeline_job(job_id, status="running", current="Processing...")
    db.get_pipeline_job(job_id)
    db.list_pipeline_jobs()

    # Batch jobs
    db.create_batch_job(batch_id, total=10)
    db.update_batch_job(batch_id, completed=5, current="5/10 done")
    db.get_batch_job(batch_id)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "output_data" / "jobs.db"

# Schema version for future migrations
SCHEMA_VERSION = 1

_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create the database tables if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_jobs (
            job_id TEXT PRIMARY KEY,
            video_path TEXT NOT NULL,
            video_name TEXT NOT NULL,
            shot_type TEXT NOT NULL DEFAULT 'unknown',
            digest TEXT,
            steps TEXT NOT NULL DEFAULT '[]',
            current TEXT NOT NULL DEFAULT 'Queued…',
            done INTEGER NOT NULL DEFAULT 0,
            ok INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            started_at REAL,
            finished_at REAL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batch_jobs (
            batch_id TEXT PRIMARY KEY,
            total INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0,
            results TEXT NOT NULL DEFAULT '[]',
            current TEXT NOT NULL DEFAULT 'Starting…',
            done INTEGER NOT NULL DEFAULT 0,
            ok INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            started_at REAL,
            finished_at REAL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    # Record schema version
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION))
    )
    conn.commit()


def close_db():
    """Close the thread-local connection."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


# ---------------------------------------------------------------------------
# Pipeline jobs
# ---------------------------------------------------------------------------

def create_pipeline_job(job_id: str, video_path: str, video_name: str,
                        shot_type: str = "unknown", digest: str = None):
    """Insert a new pipeline job into the database."""
    conn = _get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO pipeline_jobs
           (job_id, video_path, video_name, shot_type, digest, steps, current,
            done, ok, error, started_at, finished_at, created_at)
           VALUES (?, ?, ?, ?, ?, '[]', 'Queued…', 0, 0, NULL, NULL, NULL, ?)""",
        (job_id, video_path, video_name, shot_type, digest, time.time())
    )
    conn.commit()


def update_pipeline_job(job_id: str, **kwargs):
    """Update fields on a pipeline job.

    Common kwargs: status, current, done, ok, error, steps, video_name,
    started_at, finished_at
    """
    if not kwargs:
        return
    conn = _get_connection()
    # Map Python-friendly keys to DB columns
    key_map = {
        "status": "current",
        "current": "current",
        "done": "done",
        "ok": "ok",
        "error": "error",
        "steps": "steps",
        "video_name": "video_name",
        "started_at": "started_at",
        "finished_at": "finished_at",
    }
    sets = []
    values = []
    for k, v in kwargs.items():
        col = key_map.get(k, k)
        if col == "steps" and isinstance(v, list):
            v = json.dumps(v)
        if col in ("done", "ok"):
            v = int(v)
        sets.append(f"{col} = ?")
        values.append(v)
    if not sets:
        return
    values.append(job_id)
    conn.execute(
        f"UPDATE pipeline_jobs SET {', '.join(sets)} WHERE job_id = ?",
        values
    )
    conn.commit()


def get_pipeline_job(job_id: str) -> Optional[Dict]:
    """Retrieve a pipeline job by ID, or None if not found."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_pipeline_dict(row)


def list_pipeline_jobs(limit: int = 100) -> List[Dict]:
    """List pipeline jobs, newest first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM pipeline_jobs ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [_row_to_pipeline_dict(r) for r in rows]


def delete_pipeline_job(job_id: str):
    """Delete a pipeline job."""
    conn = _get_connection()
    conn.execute("DELETE FROM pipeline_jobs WHERE job_id = ?", (job_id,))
    conn.commit()


def prune_old_pipeline_jobs(max_age_seconds: int = 86400):
    """Delete jobs older than the given age."""
    cutoff = time.time() - max_age_seconds
    conn = _get_connection()
    conn.execute(
        "DELETE FROM pipeline_jobs WHERE done = 1 AND finished_at < ?",
        (cutoff,)
    )
    conn.commit()


def _row_to_pipeline_dict(row: sqlite3.Row) -> Dict:
    """Convert a sqlite3.Row to a pipeline job dict."""
    return {
        "job_id": row["job_id"],
        "video_path": row["video_path"],
        "video_name": row["video_name"],
        "shot_type": row["shot_type"],
        "digest": row["digest"],
        "steps": json.loads(row["steps"]) if row["steps"] else [],
        "current": row["current"],
        "done": bool(row["done"]),
        "ok": bool(row["ok"]),
        "error": row["error"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Batch jobs
# ---------------------------------------------------------------------------

def create_batch_job(batch_id: str, total: int = 0):
    """Insert a new batch job into the database."""
    conn = _get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO batch_jobs
           (batch_id, total, completed, results, current, done, ok, error,
            started_at, finished_at, created_at)
           VALUES (?, ?, 0, '[]', 'Starting…', 0, 0, NULL, NULL, NULL, ?)""",
        (batch_id, total, time.time())
    )
    conn.commit()


def update_batch_job(batch_id: str, **kwargs):
    """Update fields on a batch job."""
    if not kwargs:
        return
    conn = _get_connection()
    key_map = {
        "total": "total",
        "completed": "completed",
        "results": "results",
        "current": "current",
        "done": "done",
        "ok": "ok",
        "error": "error",
        "started_at": "started_at",
        "finished_at": "finished_at",
    }
    sets = []
    values = []
    for k, v in kwargs.items():
        col = key_map.get(k, k)
        if col == "results" and isinstance(v, list):
            v = json.dumps(v)
        if col in ("done", "ok"):
            v = int(v)
        sets.append(f"{col} = ?")
        values.append(v)
    if not sets:
        return
    values.append(batch_id)
    conn.execute(
        f"UPDATE batch_jobs SET {', '.join(sets)} WHERE batch_id = ?",
        values
    )
    conn.commit()


def get_batch_job(batch_id: str) -> Optional[Dict]:
    """Retrieve a batch job by ID, or None if not found."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM batch_jobs WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_batch_dict(row)


def list_batch_jobs(limit: int = 100) -> List[Dict]:
    """List batch jobs, newest first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [_row_to_batch_dict(r) for r in rows]


def delete_batch_job(batch_id: str):
    """Delete a batch job."""
    conn = _get_connection()
    conn.execute("DELETE FROM batch_jobs WHERE batch_id = ?", (batch_id,))
    conn.commit()


def _row_to_batch_dict(row: sqlite3.Row) -> Dict:
    """Convert a sqlite3.Row to a batch job dict."""
    return {
        "batch_id": row["batch_id"],
        "total": row["total"],
        "completed": row["completed"],
        "results": json.loads(row["results"]) if row["results"] else [],
        "current": row["current"],
        "done": bool(row["done"]),
        "ok": bool(row["ok"]),
        "error": row["error"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Stats / maintenance
# ---------------------------------------------------------------------------

def get_stats() -> Dict:
    """Return database statistics."""
    conn = _get_connection()
    pipeline_count = conn.execute(
        "SELECT COUNT(*) FROM pipeline_jobs"
    ).fetchone()[0]
    batch_count = conn.execute(
        "SELECT COUNT(*) FROM batch_jobs"
    ).fetchone()[0]
    return {
        "pipeline_jobs": pipeline_count,
        "batch_jobs": batch_count,
        "db_path": str(DB_PATH),
        "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
    }


# Initialise on import so the DB is always ready
init_db()
