"""Coordinator-side render orchestration shared by the local worker
(app.render.local_worker) and the remote-worker HTTP API
(app.api.routes_worker) -- both need to: claim a queued render job, load
that video's telemetry, and run the (cheap) prepare step, which only the
coordinator can do since it's the only side with the original upload.

`work_dir` for a render job is always `settings.work_dir / job_id` by
convention -- this lets `/api/worker/jobs/{id}/inputs/*` locate a claimed
job's prepared files (the trimmed clip, the telemetry manifest) from a
later, separate HTTP request with no in-memory state to carry over.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.core.config import Settings
from app.core.jobs import JobManager, JobRecord
from app.core.video_store import VideoStore
from app.render.video_render import PreparedRenderJob, prepare_render_job

TELEMETRY_MANIFEST_NAME = "telemetry.json"
TRIMMED_VIDEO_NAME = "trimmed.mp4"


class RenderPrepFailed(Exception):
    """Raised when a render job was successfully claimed but couldn't be
    prepared (e.g. its video expired between upload and claim). The job has
    already been marked `error` by the time this is raised -- callers
    should treat it as "we did handle a job" and loop for more work, not as
    "the queue is empty"."""


def work_dir_for_job(settings: Settings, job_id: str) -> Path:
    return settings.work_dir / job_id


def write_telemetry_manifest(prepared: PreparedRenderJob) -> Path:
    manifest_path = prepared.work_dir / TELEMETRY_MANIFEST_NAME
    manifest_path.write_text(json.dumps({
        "points": prepared.windowed_telemetry.to_dict(orient="records"),
        "lap_indices": prepared.lap_indices,
        "fps": prepared.fps,
    }))
    return manifest_path


def _prepare_claimed_job(job: JobRecord, job_manager: JobManager, video_store: VideoStore, settings: Settings) -> PreparedRenderJob:
    """Runs the prep step for an already-claimed job (loads its telemetry,
    trims the video, windows the telemetry, writes the manifest). Shared by
    both claim paths below -- claiming is the only part that differs
    between "next in queue" and "this specific job".

    Raises RenderPrepFailed (after already marking the job `error`) if the
    job's video is no longer available -- callers should catch this and
    treat it as "we did handle a job", not "nothing to do".
    """
    video_id = job.payload["video_id"]
    meta = video_store.get(video_id)
    if meta is None:
        job_manager.mark_error(job.id, "video no longer exists (it may have expired -- see retention policy)")
        raise RenderPrepFailed(job.id)

    if meta.laps_ready and video_store.laps_annotated_path(video_id).exists():
        df = pd.read_parquet(video_store.laps_annotated_path(video_id))
        lap_indices = video_store.load_lap_indices(video_id)
    else:
        df = pd.read_parquet(video_store.telemetry_path(video_id))
        df["lap"] = 0
        df["last_lap_s"] = 0.0
        df["lap_elapsed_s"] = 0.0
        lap_indices = []

    work_dir = work_dir_for_job(settings, job.id)
    try:
        prepared = prepare_render_job(
            Path(meta.video_path), df, lap_indices,
            job.payload["trim_start"], job.payload["trim_end"], work_dir,
            # trim_video re-encodes the (possibly large) original upload and
            # can itself run long enough to matter for the lease clock --
            # see StaleRenderJobRequeuer. Heartbeat it the same way the
            # render phase already heartbeats via job_manager.update_progress.
            on_progress=lambda p: job_manager.update_progress(job.id, p),
        )
    except Exception as exc:
        job_manager.mark_error(job.id, f"failed to prepare render: {exc}")
        raise RenderPrepFailed(job.id) from exc

    write_telemetry_manifest(prepared)
    return prepared


def claim_and_prepare_render(
    job_manager: JobManager,
    video_store: VideoStore,
    settings: Settings,
    worker_id: str,
) -> tuple[JobRecord, PreparedRenderJob] | None:
    """Claims the next queued render job (if any) and runs the cheap
    trim+window prep for it. Returns None if the queue is empty (including
    "every pending job is still awaiting an explicit self-render/release
    decision" -- see `JobManager.claim_next`'s `released` requirement). See
    `_prepare_claimed_job` for the RenderPrepFailed contract."""
    job = job_manager.claim_next("render", worker_id)
    if job is None:
        return None
    return job, _prepare_claimed_job(job, job_manager, video_store, settings)


def claim_and_prepare_specific_render(
    job_manager: JobManager,
    video_store: VideoStore,
    settings: Settings,
    job_id: str,
    worker_id: str,
) -> tuple[JobRecord, PreparedRenderJob] | None:
    """Claims one exact job (used for job-scoped self-render tokens --
    see app.api.routes_worker's /jobs/{id}/claim). Returns None if it's not
    available to claim (already claimed/done/doesn't exist)."""
    job = job_manager.claim_specific(job_id, worker_id)
    if job is None:
        return None
    return job, _prepare_claimed_job(job, job_manager, video_store, settings)
