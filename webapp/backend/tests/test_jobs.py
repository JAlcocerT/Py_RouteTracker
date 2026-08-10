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
