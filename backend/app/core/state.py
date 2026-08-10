"""Process-wide singletons. This is a single-process local app -- no DI
container needed, just one place these are constructed so API routes and
tests can import them consistently."""

from __future__ import annotations

from app.core.config import settings
from app.core.jobs import JobManager
from app.core.video_store import VideoStore

job_manager = JobManager(settings.jobs_db_path)
video_store = VideoStore(settings.upload_dir, settings.cache_dir)
