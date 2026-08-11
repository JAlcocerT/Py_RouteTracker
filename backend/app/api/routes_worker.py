"""HTTP API for remote render workers (app.worker_main). Every route is
gated by `require_worker_token`: hard-disabled (503) unless
ROUTETRACKER_WORKER_TOKEN is set, and requires that exact bearer token
otherwise. See backend/readme.md's "Distributed rendering" section for the
security model -- this is a shared-secret trust model (anyone holding the
token can claim, and briefly receive, any pending render on this instance),
appropriate for a homelab + trusted friends, not a public service.
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
    work_dir_for_job,
)

router = APIRouter(prefix="/api/worker", tags=["worker"])


def require_worker_token(authorization: str | None = Header(None)) -> None:
    if not settings.worker_token:
        raise HTTPException(503, "Distributed rendering is not enabled on this server (ROUTETRACKER_WORKER_TOKEN is not set).")
    if authorization != f"Bearer {settings.worker_token}":
        raise HTTPException(401, "Missing or invalid worker token.")


@router.get("/jobs/next")
async def claim_next_job(worker_id: str, _auth: None = Depends(require_worker_token)):
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


@router.get("/jobs/{job_id}/inputs/video")
async def get_input_video(job_id: str, _auth: None = Depends(require_worker_token)):
    video_path = work_dir_for_job(settings, job_id) / "trimmed.mp4"
    if not video_path.exists():
        raise HTTPException(404, "No prepared input video for this job (already completed, expired, or never claimed).")
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/jobs/{job_id}/inputs/telemetry")
async def get_input_telemetry(job_id: str, _auth: None = Depends(require_worker_token)):
    manifest_path = work_dir_for_job(settings, job_id) / TELEMETRY_MANIFEST_NAME
    if not manifest_path.exists():
        raise HTTPException(404, "No prepared telemetry for this job (already completed, expired, or never claimed).")
    return json.loads(manifest_path.read_text())


class ProgressUpdate(BaseModel):
    progress: float


@router.post("/jobs/{job_id}/progress")
async def report_progress(job_id: str, body: ProgressUpdate, _auth: None = Depends(require_worker_token)):
    if job_manager.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    job_manager.update_progress(job_id, body.progress)
    return {"ok": True}


@router.post("/jobs/{job_id}/complete")
async def complete_job(job_id: str, file: UploadFile, _auth: None = Depends(require_worker_token)):
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
async def fail_job(job_id: str, body: FailureReport, _auth: None = Depends(require_worker_token)):
    if job_manager.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    shutil.rmtree(work_dir_for_job(settings, job_id), ignore_errors=True)
    job_manager.mark_error(job_id, body.error)
    return {"ok": True}
