"""A deliberately small background-job manager.

This is a single-user, local-self-hosted tool -- a full Celery/Redis stack
would be operational weight with no payoff here. Instead: a thread pool runs
job functions (each job function is free to spawn its own multiprocessing
pool, e.g. for parallel HUD-frame rendering -- see app.render.video_render),
and job status is mirrored into a small SQLite table so it survives a
backend restart.

Render jobs additionally go through a pull-based claim queue (`enqueue` /
`claim_next` / `requeue_stale`) so they can be picked up either by the
built-in local worker (app.render.local_worker) or a remote one (see
app.worker_main and app.api.routes_worker) -- see the "Distributed render
workers" plan for the full design. Extraction jobs don't use this; they
still go straight through `submit`, unchanged.
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
    payload: dict[str, Any] | None = None
    worker_id: str | None = None
    claim_token: str | None = None


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
            # Added for the render job queue -- guarded rather than assumed,
            # so an existing jobs.db from before this feature keeps working.
            existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            for column in ("payload", "worker_id", "claim_token"):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")

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
        return self._row_to_record(row)

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            progress=row["progress"],
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            payload=json.loads(row["payload"]) if row["payload"] else None,
            worker_id=row["worker_id"],
            claim_token=row["claim_token"],
        )

    def _set(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        worker_id: str | None = None,
        claim_token: str | None = None,
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
        if payload is not None:
            fields.append("payload = ?"); values.append(json.dumps(payload))
        if worker_id is not None:
            fields.append("worker_id = ?"); values.append(worker_id)
        if claim_token is not None:
            fields.append("claim_token = ?"); values.append(claim_token)
        fields.append("updated_at = ?"); values.append(_now_iso())
        values.append(job_id)

        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def update_progress(self, job_id: str, progress: float) -> None:
        """Public counterpart to the private progress-reporting `submit()`
        does internally -- used by routes_worker.py, where the progress
        update comes from an HTTP request (a remote worker), not a local
        closure. Also serves as the claim's heartbeat: `requeue_stale`
        checks this same `updated_at` bump."""
        self._set(job_id, progress=progress)

    def mark_done(self, job_id: str, result: dict[str, Any]) -> None:
        self._set(job_id, status="done", progress=1.0, result=result)

    def mark_error(self, job_id: str, error: str) -> None:
        self._set(job_id, status="error", error=error)

    def enqueue(self, job_id: str, payload: dict[str, Any], claim_token: str | None = None) -> None:
        """Marks a job as queued/claimable, with the data a worker (local or
        remote) needs to actually execute it. Unlike `submit`, this does NOT
        run anything -- the job sits `pending` until some worker calls
        `claim_next` or (given the matching `claim_token`) `claim_specific`.
        """
        self._set(job_id, payload=payload, claim_token=claim_token)

    def claim_specific(self, job_id: str, worker_id: str) -> JobRecord | None:
        """Claims one exact job by id, if it's still `pending` -- None if
        it's already claimed/done/doesn't exist. Used for job-scoped
        self-render tokens, where the caller already knows exactly which
        job it wants (unlike `claim_next`'s "whatever's oldest"). Same
        atomicity guarantee: the whole select-then-update happens under
        `self._lock`."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND status = 'pending' AND payload IS NOT NULL",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            now = _now_iso()
            conn.execute(
                "UPDATE jobs SET status = 'running', worker_id = ?, updated_at = ? WHERE id = ?",
                (worker_id, now, job_id),
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_record(claimed)

    def claim_next(self, kind: str, worker_id: str, min_age_seconds: float = 0) -> JobRecord | None:
        """Atomically claims the oldest unclaimed `pending` job of `kind`,
        or returns None if there isn't one. Safe to call concurrently from
        multiple workers (local and/or remote) -- the whole
        select-then-update happens under `self._lock`, so two callers can
        never claim the same job.

        `min_age_seconds`, if given, excludes jobs younger than that from
        consideration -- this is the per-upload self-render grace period
        (see app.render.coordinator): a brand-new render job is invisible
        to "claim whatever's next" (the built-in local worker, and any
        standing remote workers) for a short window, so the uploader has a
        real chance to claim it themselves first via `claim_specific`
        (which has no such delay). Without this, the built-in worker's
        ~2s poll loop wins the race almost every time, making the
        self-render option effectively unusable.
        """
        with self._lock, self._connect() as conn:
            query = "SELECT * FROM jobs WHERE kind = ? AND status = 'pending' AND payload IS NOT NULL"
            params: list[Any] = [kind]
            if min_age_seconds > 0:
                query += " AND created_at <= ?"
                params.append(_iso_minus_seconds(min_age_seconds))
            query += " ORDER BY created_at ASC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row is None:
                return None
            now = _now_iso()
            conn.execute(
                "UPDATE jobs SET status = 'running', worker_id = ?, updated_at = ? WHERE id = ?",
                (worker_id, now, row["id"]),
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return self._row_to_record(claimed)

    def requeue_stale(self, kind: str, lease_seconds: float) -> int:
        """Resets any `running` job of `kind` whose worker hasn't reported
        progress (bumped `updated_at`) in over `lease_seconds` back to
        `pending`, so another worker can pick it up -- covers a worker that
        vanished mid-render (closed laptop, network drop, crash). Returns
        how many were reset."""
        cutoff = _iso_minus_seconds(lease_seconds)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'pending', worker_id = NULL "
                "WHERE kind = ? AND status = 'running' AND worker_id IS NOT NULL AND updated_at < ?",
                (kind, cutoff),
            )
            return cur.rowcount


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _iso_minus_seconds(seconds: float) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
