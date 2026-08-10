"""A deliberately small background-job manager.

This is a single-user, local-self-hosted tool -- a full Celery/Redis stack
would be operational weight with no payoff here. Instead: a thread pool runs
job functions (each job function is free to spawn its own multiprocessing
pool, e.g. for parallel HUD-frame rendering -- see app.render.video_render),
and job status is mirrored into a small SQLite table so it survives a
backend restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

JobFn = Callable[[Callable[[float], None]], dict[str, Any]]


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str  # pending | running | done | error
    progress: float
    error: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str


class JobManager:
    def __init__(self, db_path: Path, max_workers: int = 4):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_job(self, kind: str) -> str:
        job_id = str(uuid.uuid4())
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, status, progress, error, result, created_at, updated_at) "
                "VALUES (?, ?, 'pending', 0, NULL, NULL, ?, ?)",
                (job_id, kind, now, now),
            )
        return job_id

    def submit(self, job_id: str, fn: JobFn) -> None:
        self._set(job_id, status="running")

        def run() -> None:
            try:
                result = fn(lambda p: self._set(job_id, progress=p))
                self._set(job_id, status="done", progress=1.0, result=result)
            except Exception as exc:  # noqa: BLE001 - job errors must not crash the worker thread
                self._set(job_id, status="error", error=f"{exc}\n{traceback.format_exc()}")

        self._executor.submit(run)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            progress=row["progress"],
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _set(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        fields, values = [], []
        if status is not None:
            fields.append("status = ?"); values.append(status)
        if progress is not None:
            fields.append("progress = ?"); values.append(progress)
        if error is not None:
            fields.append("error = ?"); values.append(error)
        if result is not None:
            fields.append("result = ?"); values.append(json.dumps(result))
        fields.append("updated_at = ?"); values.append(_now_iso())
        values.append(job_id)

        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
