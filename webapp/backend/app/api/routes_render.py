from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.state import job_manager, video_store
from app.render.hud_layers import RenderConfig
from app.render.video_render import render_and_composite

router = APIRouter(prefix="/api", tags=["render"])


class WidgetSelection(BaseModel):
    speedo: bool = True
    gg: bool = True
    minimap: bool = True
    session_graph: bool = True


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

    if meta.laps_ready and video_store.laps_annotated_path(video_id).exists():
        df = pd.read_parquet(video_store.laps_annotated_path(video_id))
        lap_indices = video_store.load_lap_indices(video_id)
    else:
        df = pd.read_parquet(video_store.telemetry_path(video_id))
        df["lap"] = 0
        df["last_lap_s"] = 0.0
        lap_indices = []

    config = RenderConfig(
        enable_speedo=body.widgets.speedo,
        enable_gg=body.widgets.gg,
        enable_minimap=body.widgets.minimap,
        enable_session_graph=body.widgets.session_graph,
        max_expected_speed_kmh=body.style.max_expected_speed_kmh,
        limit_g=body.style.limit_g,
        theme=body.style.theme,
    )

    job_id = job_manager.create_job("render")
    output_filename = f"{video_id}_{job_id}.mp4"
    output_path = settings.output_dir / output_filename
    work_dir = settings.work_dir / job_id

    def render(progress_cb):
        render_and_composite(
            source_video=Path(meta.video_path),
            annotated_telemetry=df,
            lap_indices=lap_indices,
            config=config,
            trim_start=body.trim_start,
            trim_end=body.trim_end,
            work_dir=work_dir,
            output_path=output_path,
            n_workers=settings.max_render_workers,
            on_progress=progress_cb,
        )
        return {"output_file": str(output_path), "download_url": f"/api/render/{job_id}/download"}

    job_manager.submit(job_id, render)
    return {"job_id": job_id}


@router.get("/render/{job_id}/download")
async def download_render(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(400, f"render job is not finished (status={job.status})")
    output_file = job.result.get("output_file") if job.result else None
    if not output_file or not Path(output_file).exists():
        raise HTTPException(410, "rendered file is no longer available")
    return FileResponse(output_file, media_type="video/mp4", filename=Path(output_file).name)
