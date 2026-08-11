from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.state import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "result": job.result,
        # "local" for the built-in worker, "self" for a job-scoped
        # self-render, a client-supplied name for a persistent remote
        # worker (see app.worker_main --name), or None if unclaimed / not a
        # queued render job (e.g. an extraction job).
        "worker_id": job.worker_id,
        # Present only for queued render jobs -- lets the frontend show a
        # "render this yourself" command while status is still "pending".
        # This app has no per-resource auth anywhere else either (job/video
        # UUIDs are already the only access control that exists), so
        # exposing it here doesn't change the security posture.
        "claim_token": job.claim_token,
        # Whether this job has been released for general claiming (see
        # POST /jobs/{id}/release) -- a still-pending, unreleased render job
        # is waiting on the uploader's explicit decision (self-render vs.
        # "render on the server"), not stuck.
        "released": job.released,
    }


@router.post("/{job_id}/release")
async def release_job(job_id: str):
    """The uploader's explicit "no, just render this on the server"
    decision -- makes the job eligible for `claim_next` (the built-in
    local worker, or any standing remote worker). Doesn't render anything
    itself; some worker still has to actually poll and claim it, same as
    always. See app.core.jobs.JobManager.release."""
    if job_manager.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    job_manager.release(job_id)
    return {"ok": True}
