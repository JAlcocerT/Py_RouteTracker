"""The built-in worker that makes local-only operation keep working with
zero configuration -- a background thread that claims and executes render
jobs from the same queue a remote worker (app.worker_main) would use, so
"no remote workers configured" behaves exactly like "one worker: this
machine". Unlike the old code (a ThreadPoolExecutor that could run several
renders concurrently, each spawning its own multiprocessing pool), this
processes jobs strictly one at a time -- a more predictable ceiling on
constrained/shared hardware.
"""

from __future__ import annotations

import logging
import shutil
import threading
import traceback

from app.core.config import Settings
from app.core.jobs import JobManager
from app.core.video_store import VideoStore
from app.render.coordinator import RenderPrepFailed, claim_and_prepare_render
from app.render.render_config import render_config_from_payload
from app.render.video_render import execute_prepared_render

logger = logging.getLogger(__name__)


class LocalRenderWorker:
    def __init__(self, job_manager: JobManager, video_store: VideoStore, settings: Settings, poll_interval: float = 2.0):
        self._job_manager = job_manager
        self._video_store = video_store
        self._settings = settings
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="local-render-worker")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            processed = self._process_one()
            if not processed:
                self._stop.wait(self._poll_interval)

    def _process_one(self) -> bool:
        """Returns True if a job was claimed (whether it succeeded, failed
        to render, or failed to even prepare) -- False only means the queue
        was genuinely empty, which is when the caller should back off."""
        try:
            claimed = claim_and_prepare_render(self._job_manager, self._video_store, self._settings, worker_id="local")
        except RenderPrepFailed:
            return True
        if claimed is None:
            return False

        job, prepared = claimed
        result, error = None, None
        try:
            config = render_config_from_payload(job.payload)
            output_path = self._settings.output_dir / f"{job.payload['video_id']}_{job.id}.mp4"
            execute_prepared_render(
                prepared, config, output_path,
                n_workers=self._settings.max_render_workers,
                on_progress=lambda p: self._job_manager.update_progress(job.id, p),
            )
            result = {"output_file": str(output_path), "video_id": job.payload["video_id"]}
        except Exception as exc:  # noqa: BLE001 - job errors must not kill this thread
            error = f"{exc}\n{traceback.format_exc()}"

        # A render on a large enough video can run past this job's lease
        # (see StaleRenderJobRequeuer), in which case someone else may have
        # already reclaimed and re-rendered it while this render was still
        # running. app.api.routes_worker._require_current_lease guards
        # exactly this for remote workers; this is the same check for the
        # built-in local one, which has no HTTP round-trip to hang it off
        # of. Skip straight to returning -- neither mark_done/mark_error nor
        # the work_dir cleanup below are safe here, since a new claimant may
        # already be using this same (job-id-keyed) work_dir.
        current = self._job_manager.get_job(job.id)
        if current is None or current.lease_id != job.lease_id:
            logger.warning("Render job %s was reassigned while this local worker was still processing it -- discarding this result.", job.id)
            return True

        # trimmed.mp4 + HUD frame PNGs + telemetry.json: pure scratch,
        # zero value once the final output exists (or failed to). Done
        # BEFORE marking the job done/error below -- otherwise a client
        # polling status could see "done" and check for the (still
        # briefly present) work_dir before this cleanup has run.
        shutil.rmtree(prepared.work_dir, ignore_errors=True)

        if error is None:
            self._job_manager.mark_done(job.id, result)
        else:
            self._job_manager.mark_error(job.id, error)
        return True


class StaleRenderJobRequeuer:
    """A render job claimed by a worker (local or remote) that vanished
    mid-render -- closed laptop, network drop, crash -- would otherwise sit
    `running` forever with nobody processing it. This periodically resets
    any render job whose last progress update is older than the lease back
    to `pending`, so another worker picks it up."""

    def __init__(self, job_manager: JobManager, lease_minutes: float, interval_seconds: float = 120):
        self._job_manager = job_manager
        self._lease_seconds = lease_minutes * 60
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="stale-render-job-requeuer")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._job_manager.requeue_stale("render", self._lease_seconds)
            except Exception:
                logger.exception("Stale render job requeue pass failed")
