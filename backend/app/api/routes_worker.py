"""HTTP API for render workers (app.worker_main). Two independent auth
paths, both bearer-token:

- `require_global_worker_token` -- the admin-configured, always-on-if-set
  shared secret (ROUTETRACKER_WORKER_TOKEN). Grants access to *any* pending
  render job. Used only by GET /jobs/next ("claim whatever's oldest"),
  since that only makes sense for a trusted standing worker. Hard-disabled
  (503) if the token isn't set -- this mechanism is opt-in.
- `require_job_access` -- accepts EITHER the global token OR one specific
  job's own `claim_token` (generated per render job in routes_render.py,
  shown in the UI to whoever just uploaded that video). This is what lets
  someone render their own upload on their own other device with **zero
  server configuration** -- no admin secret needed at all. Used by every
  other route below, once a job_id is in scope.

See backend/readme.md's "Distributed rendering" section for the full
security model.
"""

from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.core.state import job_manager, settings, video_store
from app.render.coordinator import (
    TELEMETRY_MANIFEST_NAME,
    RenderPrepFailed,
    claim_and_prepare_render,
    claim_and_prepare_specific_render,
    work_dir_for_job,
)

router = APIRouter(prefix="/api/worker", tags=["worker"])


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):]


def require_global_worker_token(authorization: str | None = Header(None)) -> None:
    if not settings.worker_token:
        raise HTTPException(503, "Distributed rendering is not enabled on this server (ROUTETRACKER_WORKER_TOKEN is not set).")
    if _bearer_token(authorization) != settings.worker_token:
        raise HTTPException(401, "Missing or invalid worker token.")


def require_job_access(job_id: str, authorization: str | None = Header(None)) -> None:
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(401, "Missing bearer token.")
    if settings.worker_token and token == settings.worker_token:
        return
    job = job_manager.get_job(job_id)
    if job is not None and job.claim_token and token == job.claim_token:
        return
    raise HTTPException(401, "Invalid token for this job.")


@router.get("/jobs/next")
async def claim_next_job(worker_id: str, _auth: None = Depends(require_global_worker_token)):
    try:
        claimed = claim_and_prepare_render(job_manager, video_store, settings, worker_id=worker_id)
    except RenderPrepFailed:
        # a job existed but couldn't be prepared -- it's already marked
        # error; tell this worker to just ask again shortly
        return Response(status_code=204)
    if claimed is None:
        return Response(status_code=204)

    job, _prepared = claimed
    return {
        "job_id": job.id,
        "widgets": job.payload["widgets"],
        "style": job.payload["style"],
    }


@router.post("/jobs/{job_id}/claim")
async def claim_specific_job(job_id: str, _auth: None = Depends(require_job_access)):
    """Claims exactly this job -- used for job-scoped self-render tokens,
    where the caller (the uploader's own other device) already knows which
    job it wants. 409 if it's no longer available (already claimed by the
    built-in local worker, a persistent remote worker, or run out this same
    way already)."""
    try:
        claimed = claim_and_prepare_specific_render(job_manager, video_store, settings, job_id, worker_id="self")
    except RenderPrepFailed as exc:
        raise HTTPException(409, "This job could not be prepared (it may have expired).") from exc
    if claimed is None:
        raise HTTPException(409, "This job is no longer available to claim (it may already be rendering, or finished).")

    job, _prepared = claimed
    return {
        "job_id": job.id,
        "widgets": job.payload["widgets"],
        "style": job.payload["style"],
    }


@router.get("/jobs/{job_id}/inputs/video")
async def get_input_video(job_id: str, _auth: None = Depends(require_job_access)):
    video_path = work_dir_for_job(settings, job_id) / "trimmed.mp4"
    if not video_path.exists():
        raise HTTPException(404, "No prepared input video for this job (already completed, expired, or never claimed).")
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/jobs/{job_id}/inputs/telemetry")
async def get_input_telemetry(job_id: str, _auth: None = Depends(require_job_access)):
    manifest_path = work_dir_for_job(settings, job_id) / TELEMETRY_MANIFEST_NAME
    if not manifest_path.exists():
        raise HTTPException(404, "No prepared telemetry for this job (already completed, expired, or never claimed).")
    return json.loads(manifest_path.read_text())


class ProgressUpdate(BaseModel):
    progress: float


@router.post("/jobs/{job_id}/progress")
async def report_progress(job_id: str, body: ProgressUpdate, _auth: None = Depends(require_job_access)):
    if job_manager.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    job_manager.update_progress(job_id, body.progress)
    return {"ok": True}


@router.post("/jobs/{job_id}/complete")
async def complete_job(job_id: str, file: UploadFile, _auth: None = Depends(require_job_access)):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    video_id = job.payload["video_id"]
    output_path = settings.output_dir / f"{video_id}_{job_id}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Cleanup before mark_done: otherwise a client polling status could see
    # "done" and check for the (still briefly present) work_dir before this
    # has run.
    shutil.rmtree(work_dir_for_job(settings, job_id), ignore_errors=True)
    job_manager.mark_done(job_id, {"output_file": str(output_path), "video_id": video_id})
    return {"ok": True}


class FailureReport(BaseModel):
    error: str


@router.post("/jobs/{job_id}/fail")
async def fail_job(job_id: str, body: FailureReport, _auth: None = Depends(require_job_access)):
    if job_manager.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    shutil.rmtree(work_dir_for_job(settings, job_id), ignore_errors=True)
    job_manager.mark_error(job_id, body.error)
    return {"ok": True}
