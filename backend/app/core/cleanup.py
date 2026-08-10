"""Automatic cleanup of uploaded/processed video files.

This is a local telemetry-overlay tool, not a video hosting service --
uploaded source videos and rendered output are only ever meant to live on
disk long enough for the user to download their result. A background
sweeper deletes each video's entire on-disk footprint (source video, GPX,
extracted-telemetry cache, lap-detection cache, rendered output) once it's
older than `settings.retention_minutes`, so nothing accumulates on the host
indefinitely.

Per-render intermediate scratch (the trimmed clip + HUD frame PNGs in
`work_dir`) is unrelated to this module -- that's deleted immediately after
each render job finishes, in routes_render.py, since it has zero value the
moment the final composited video exists.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import datetime

from app.core.config import Settings
from app.core.jobs import JobManager
from app.core.video_store import VideoStore

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = {"pending", "running"}


def _is_job_active(job_manager: JobManager, job_id: str | None) -> bool:
    if not job_id:
        return False
    job = job_manager.get_job(job_id)
    return job is not None and job.status in _ACTIVE_JOB_STATUSES


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def delete_video_artifacts(video_store: VideoStore, settings: Settings, video_id: str) -> None:
    """Removes every file a video produced: the uploaded source video/GPX
    (and its containing directory, including meta.json), cached
    telemetry/lap parquet files, and any rendered output(s). Idempotent --
    safe to call even if some paths are already gone."""
    shutil.rmtree(video_store.upload_dir / video_id, ignore_errors=True)

    for path in (
        video_store.telemetry_path(video_id),
        video_store.laps_annotated_path(video_id),
        video_store.laps_table_path(video_id),
        video_store.laps_indices_path(video_id),
    ):
        path.unlink(missing_ok=True)

    for output in settings.output_dir.glob(f"{video_id}_*.mp4"):
        output.unlink(missing_ok=True)

    logger.info("Cleaned up temporary storage for video %s", video_id)


def sweep_expired_videos(video_store: VideoStore, settings: Settings, job_manager: JobManager) -> int:
    """Deletes every video whose retention window has elapsed. Returns how
    many were cleaned up. A video is skipped (and retried on the next
    sweep) if its extraction job or any of its render jobs are still
    pending/running, so an in-progress job's files are never deleted out
    from under it."""
    if not video_store.upload_dir.exists():
        return 0

    cutoff = time.time() - settings.retention_minutes * 60
    cleaned = 0

    for video_dir in video_store.upload_dir.iterdir():
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        meta = video_store.get(video_id)
        if meta is None:
            continue

        if _is_job_active(job_manager, meta.extraction_job_id):
            continue
        if any(_is_job_active(job_manager, job_id) for job_id in meta.render_job_ids):
            continue

        created = _parse_timestamp(meta.created_at)
        if created is None:
            created = video_dir.stat().st_mtime
        if created > cutoff:
            continue

        delete_video_artifacts(video_store, settings, video_id)
        cleaned += 1

    return cleaned


class RetentionSweeper:
    """Runs `sweep_expired_videos` on a background thread every
    `interval_seconds`, for the lifetime of the process."""

    def __init__(self, video_store: VideoStore, settings: Settings, job_manager: JobManager, interval_seconds: float = 300):
        self._video_store = video_store
        self._settings = settings
        self._job_manager = job_manager
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="retention-sweeper")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # Wait first: nothing can have expired within `interval_seconds` of
        # process startup, and this avoids doing disk I/O on every reload
        # during development.
        while not self._stop.wait(self._interval):
            try:
                sweep_expired_videos(self._video_store, self._settings, self._job_manager)
            except Exception:
                logger.exception("Retention sweep failed")
