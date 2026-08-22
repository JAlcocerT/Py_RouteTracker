"""Coverage for LocalRenderWorker's lease fencing -- the built-in worker's
half of the race app.api.routes_worker._require_current_lease guards for
remote workers (see that function's docstring). A render on a large enough
video can outlast the job's lease (StaleRenderJobRequeuer), so by the time
execute_prepared_render returns, some other worker may have already
reclaimed and re-rendered the same job. Without a lease check, _process_one
would unconditionally call mark_done/mark_error and stomp whatever that new
claimant is doing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.config import Settings
from app.core.jobs import JobManager
from app.core.video_store import VideoStore
from app.render.video_render import PreparedRenderJob

import app.render.local_worker as local_worker_mod
from app.render.local_worker import LocalRenderWorker


def _make_worker(tmp_path, job_manager, monkeypatch) -> tuple[LocalRenderWorker, PreparedRenderJob]:
    monkeypatch.setenv("ROUTETRACKER_DATA_DIR", str(tmp_path / "data"))
    settings = Settings()
    # Never actually touched: claim_and_prepare_render is monkeypatched in
    # every test here, so this worker's video_store is dead weight -- give
    # it working paths anyway rather than a fragile None passed through.
    video_store = VideoStore(settings.upload_dir, settings.cache_dir)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    prepared = PreparedRenderJob(
        trimmed_video_path=work_dir / "trimmed.mp4",
        windowed_telemetry=pd.DataFrame({"time": [0.0, 1.0]}),
        lap_indices=[],
        fps=30.0,
        work_dir=work_dir,
    )
    worker = LocalRenderWorker(job_manager, video_store, settings)
    return worker, prepared


def test_process_one_discards_result_if_job_reassigned_during_render(tmp_path, monkeypatch):
    job_manager = JobManager(tmp_path / "jobs.db")
    job_id = job_manager.create_job("render")
    job_manager.enqueue(job_id, {"video_id": "vid1", "trim_start": 0.0, "trim_end": 1.0})
    job_manager.release(job_id)
    claimed_job = job_manager.claim_next("render", worker_id="local")

    worker, prepared = _make_worker(tmp_path, job_manager, monkeypatch)

    monkeypatch.setattr(
        local_worker_mod, "claim_and_prepare_render",
        lambda jm, vs, settings, worker_id: (claimed_job, prepared),
    )

    def fake_execute(prepared_arg, config, output_path, n_workers=None, on_progress=None):
        # Simulate the lease expiring and another worker reclaiming this
        # exact job while this (long) render was still running.
        job_manager.requeue_stale("render", lease_seconds=-1)
        job_manager.claim_next("render", worker_id="other-worker")

    monkeypatch.setattr(local_worker_mod, "execute_prepared_render", fake_execute)

    processed = worker._process_one()

    assert processed is True
    # must NOT have been marked done by the stale worker
    current = job_manager.get_job(job_id)
    assert current.status == "running"
    assert current.worker_id == "other-worker"
    # the reclaiming worker's work_dir must survive -- the stale worker's
    # cleanup must not have deleted it out from under the new claimant
    assert prepared.work_dir.exists()


def test_process_one_marks_done_when_lease_still_current(tmp_path, monkeypatch):
    job_manager = JobManager(tmp_path / "jobs.db")
    job_id = job_manager.create_job("render")
    job_manager.enqueue(job_id, {"video_id": "vid1", "trim_start": 0.0, "trim_end": 1.0})
    job_manager.release(job_id)
    claimed_job = job_manager.claim_next("render", worker_id="local")

    worker, prepared = _make_worker(tmp_path, job_manager, monkeypatch)

    monkeypatch.setattr(
        local_worker_mod, "claim_and_prepare_render",
        lambda jm, vs, settings, worker_id: (claimed_job, prepared),
    )
    monkeypatch.setattr(local_worker_mod, "execute_prepared_render", lambda *a, **k: None)

    processed = worker._process_one()

    assert processed is True
    current = job_manager.get_job(job_id)
    assert current.status == "done"
    assert not prepared.work_dir.exists()  # scratch cleaned up
