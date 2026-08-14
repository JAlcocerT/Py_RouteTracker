from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.state import job_manager, video_store

router = APIRouter(prefix="/api", tags=["render"])


class WidgetSelection(BaseModel):
    speedo: bool = True
    gg: bool = True
    minimap: bool = True


class RenderStyle(BaseModel):
    theme: str = "cyberpunk"
    max_expected_speed_kmh: float = 85.0
    limit_g: float = 1.5


class RenderRequest(BaseModel):
    trim_start: float
    trim_end: float
    widgets: WidgetSelection = WidgetSelection()
    style: RenderStyle = RenderStyle()


@router.post("/videos/{video_id}/render")
async def start_render(video_id: str, body: RenderRequest):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    if not meta.telemetry_ready:
        raise HTTPException(425, "telemetry extraction is not finished yet")
    if body.trim_end <= body.trim_start:
        raise HTTPException(400, "trim_end must be after trim_start")

    # The actual render (trim + HUD frames + composite) doesn't happen here
    # -- this just queues it. Whoever claims it (the always-running
    # LocalRenderWorker by default, a remote worker with the global
    # ROUTETRACKER_WORKER_TOKEN, or -- via claim_token below -- the
    # uploader's own other device) does the work and reports back through
    # the same job_id, so polling /api/jobs/{id} looks identical regardless
    # of who rendered it.
    job_id = job_manager.create_job("render")
    payload = {
        "video_id": video_id,
        "trim_start": body.trim_start,
        "trim_end": body.trim_end,
        "widgets": body.widgets.model_dump(),
        "style": body.style.model_dump(),
    }
    # A one-time credential scoped to exactly this job -- lets the uploader
    # run `app.worker_main --job <id> --token <claim_token>` on their own
    # other device to render this specific video, with no server-side
    # configuration (no ROUTETRACKER_WORKER_TOKEN needed) and nothing to
    # share with anyone else. See app.api.routes_worker.require_job_access.
    claim_token = secrets.token_urlsafe(24)
    video_store.append_render_job_id(video_id, job_id)
    job_manager.enqueue(job_id, payload, claim_token=claim_token)
    return {"job_id": job_id, "claim_token": claim_token}


@router.get("/render/{job_id}/download")
async def download_render(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(400, f"render job is not finished (status={job.status})")

    result = job.result or {}
    output_file = result.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(410, "This file has expired (uploaded/rendered videos are temporary and auto-deleted after the retention window -- see /api/health).")

    video_meta = video_store.get(result.get("video_id", ""))
    base_name = Path(video_meta.filename).stem if video_meta else "overlay"
    download_name = f"{base_name}_overlay.mp4"

    return FileResponse(output_file, media_type="video/mp4", filename=download_name)
