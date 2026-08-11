import time

import pytest

from app.core.jobs import JobManager


def _wait_for_terminal(manager: JobManager, job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_job_runs_to_completion(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("test")

    assert manager.get_job(job_id).status == "pending"

    def work(progress_cb):
        progress_cb(0.5)
        return {"ok": True}

    manager.submit(job_id, work)
    job = _wait_for_terminal(manager, job_id)

    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result == {"ok": True}


def test_job_records_error(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("test")

    def failing(progress_cb):
        raise ValueError("boom")

    manager.submit(job_id, failing)
    job = _wait_for_terminal(manager, job_id)

    assert job.status == "error"
    assert "boom" in job.error


def test_get_unknown_job_returns_none(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    assert manager.get_job("does-not-exist") is None


def test_progress_updates_are_visible_before_completion(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("test")
    started = threading_event = __import__("threading").Event()
    proceed = __import__("threading").Event()

    def work(progress_cb):
        progress_cb(0.3)
        started.set()
        proceed.wait(timeout=5)
        return {"ok": True}

    manager.submit(job_id, work)
    assert started.wait(timeout=5)
    mid_job = manager.get_job(job_id)
    assert mid_job.status == "running"
    assert mid_job.progress == pytest.approx(0.3)

    proceed.set()
    job = _wait_for_terminal(manager, job_id)
    assert job.status == "done"


# --- pull-based render job queue (enqueue / claim_next / requeue_stale) ---


def test_claim_next_returns_none_when_queue_empty(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    assert manager.claim_next("render", worker_id="local") is None


def test_created_but_not_enqueued_job_is_not_claimable(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    manager.create_job("render")  # no enqueue() -- no payload yet
    assert manager.claim_next("render", worker_id="local") is None


def test_enqueue_then_claim_next_round_trip(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc", "trim_start": 0.0, "trim_end": 20.0})

    claimed = manager.claim_next("render", worker_id="friend-laptop")

    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.status == "running"
    assert claimed.worker_id == "friend-laptop"
    assert claimed.payload == {"video_id": "abc", "trim_start": 0.0, "trim_end": 20.0}


def test_claim_next_never_returns_the_same_job_twice(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"})

    first = manager.claim_next("render", worker_id="worker-a")
    second = manager.claim_next("render", worker_id="worker-b")

    assert first.id == job_id
    assert second is None


def test_claim_next_only_matches_requested_kind(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    extract_id = manager.create_job("extract_telemetry")
    manager.enqueue(extract_id, {"whatever": True})

    assert manager.claim_next("render", worker_id="local") is None


def test_claim_next_returns_oldest_queued_job_first(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    first_id = manager.create_job("render")
    manager.enqueue(first_id, {"n": 1})
    time.sleep(0.01)
    second_id = manager.create_job("render")
    manager.enqueue(second_id, {"n": 2})

    claimed = manager.claim_next("render", worker_id="local")
    assert claimed.id == first_id


def test_update_progress_mark_done_mark_error(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"})
    manager.claim_next("render", worker_id="local")

    manager.update_progress(job_id, 0.42)
    assert manager.get_job(job_id).progress == pytest.approx(0.42)

    manager.mark_done(job_id, {"output_file": "/tmp/out.mp4"})
    done = manager.get_job(job_id)
    assert done.status == "done"
    assert done.progress == 1.0
    assert done.result == {"output_file": "/tmp/out.mp4"}


def test_mark_error(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"})
    manager.claim_next("render", worker_id="local")

    manager.mark_error(job_id, "worker disconnected")
    job = manager.get_job(job_id)
    assert job.status == "error"
    assert job.error == "worker disconnected"


def test_requeue_stale_resets_a_silent_claimed_job(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"})
    manager.claim_next("render", worker_id="vanished-worker")

    # negative lease -> cutoff is in the future -> any claimed job (even one
    # claimed a moment ago) counts as stale, without needing to sleep
    reset_count = manager.requeue_stale("render", lease_seconds=-1)

    assert reset_count == 1
    job = manager.get_job(job_id)
    assert job.status == "pending"
    assert job.worker_id is None
    # requeued jobs must still be claimable again
    assert manager.claim_next("render", worker_id="another-worker").id == job_id


def test_requeue_stale_leaves_recently_updated_jobs_alone(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"})
    manager.claim_next("render", worker_id="active-worker")

    reset_count = manager.requeue_stale("render", lease_seconds=3600)

    assert reset_count == 0
    job = manager.get_job(job_id)
    assert job.status == "running"
    assert job.worker_id == "active-worker"


# --- job-scoped claim (claim_specific), for per-upload self-render tokens ---


def test_claim_specific_claims_the_requested_job(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"}, claim_token="secret-for-this-job")

    claimed = manager.claim_specific(job_id, worker_id="self")

    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.status == "running"
    assert claimed.worker_id == "self"
    assert claimed.claim_token == "secret-for-this-job"


def test_claim_specific_returns_none_for_unknown_job(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    assert manager.claim_specific("does-not-exist", worker_id="self") is None


def test_claim_specific_returns_none_once_already_claimed(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")
    manager.enqueue(job_id, {"video_id": "abc"}, claim_token="tok")

    first = manager.claim_specific(job_id, worker_id="self")
    second = manager.claim_specific(job_id, worker_id="someone-else")

    assert first is not None
    assert second is None


def test_claim_specific_does_not_disturb_other_queued_jobs(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    older_id = manager.create_job("render")
    manager.enqueue(older_id, {"video_id": "older"}, claim_token="tok-older")
    time.sleep(0.01)
    newer_id = manager.create_job("render")
    manager.enqueue(newer_id, {"video_id": "newer"}, claim_token="tok-newer")

    # claim the NEWER job specifically -- claim_next's "oldest first" must
    # not be affected, and the older job must remain claimable afterward
    claimed = manager.claim_specific(newer_id, worker_id="self")
    assert claimed.id == newer_id

    still_pending = manager.claim_next("render", worker_id="local")
    assert still_pending.id == older_id


def test_claim_specific_not_claimable_before_enqueue(tmp_path):
    manager = JobManager(tmp_path / "jobs.db")
    job_id = manager.create_job("render")  # no enqueue() -- no payload/claim_token yet
    assert manager.claim_specific(job_id, worker_id="self") is None
