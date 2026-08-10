from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.ffmpeg_utils import get_video_duration
from app.core.state import job_manager, video_store
from app.core.video_store import VideoMeta
from app.telemetry.sources.external_gpx import ExternalGpxSource
from app.telemetry.sources.gopro_embedded import GoProEmbeddedSource

router = APIRouter(prefix="/api/videos", tags=["videos"])

VALID_SOURCE_TYPES = ("gopro_embedded", "external_gpx")


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return dest


@router.post("")
async def upload_video(
    video: UploadFile,
    source_type: str = Form(...),
    gpx: Optional[UploadFile] = None,
    video_start_time: Optional[str] = Form(None),
    offset_sec: float = Form(0.0),
    target_fps: float = Form(settings.default_target_fps),
):
    if source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of {VALID_SOURCE_TYPES}")
    if source_type == "external_gpx" and gpx is None:
        raise HTTPException(400, "external_gpx source_type requires a gpx file upload")

    video_id = video_store.new_id()
    video_dir = video_store.video_dir(video_id)

    video_path = _save_upload(video, video_dir / video.filename)
    gpx_path = _save_upload(gpx, video_dir / gpx.filename) if gpx is not None else None

    try:
        duration_sec = get_video_duration(video_path)
    except Exception as exc:
        raise HTTPException(500, f"could not read video duration: {exc}") from exc

    meta = VideoMeta(
        id=video_id,
        filename=video.filename,
        video_path=str(video_path),
        source_type=source_type,
        gpx_path=str(gpx_path) if gpx_path else None,
        duration_sec=duration_sec,
    )
    video_store.save(meta)

    parsed_start_time = datetime.fromisoformat(video_start_time) if video_start_time else None

    def extract(progress_cb):
        progress_cb(0.05)
        if source_type == "gopro_embedded":
            source = GoProEmbeddedSource(cache_dir=settings.cache_dir, target_fps=target_fps)
            result = source.extract(video_path, duration_sec)
        else:
            source = ExternalGpxSource(gpx_path, target_fps=target_fps, offset_sec=offset_sec, video_start_time=parsed_start_time)
            result = source.extract(video_path, duration_sec)

        progress_cb(0.8)
        if result.df.empty:
            raise ValueError("No usable GPS telemetry found for this video/source combination")

        result.df.to_parquet(video_store.telemetry_path(video_id))
        video_store.update(video_id, telemetry_ready=True, has_accel=result.has_accel)
        progress_cb(1.0)
        return {"point_count": len(result.df), "has_accel": result.has_accel, "source_name": result.source_name}

    job_id = job_manager.create_job("extract_telemetry")
    video_store.update(video_id, extraction_job_id=job_id)
    job_manager.submit(job_id, extract)

    return {"video_id": video_id, "job_id": job_id, "duration_sec": duration_sec}


@router.get("/{video_id}")
async def get_video(video_id: str):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    return meta.to_json()


@router.get("/{video_id}/telemetry")
async def get_telemetry(video_id: str, max_points: int = 2000):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    path = video_store.telemetry_path(video_id)
    if not meta.telemetry_ready or not path.exists():
        raise HTTPException(425, "telemetry extraction is not finished yet; poll the job status")

    df = pd.read_parquet(path)
    if len(df) > max_points:
        stride = max(1, len(df) // max_points)
        df = df.iloc[::stride].reset_index(drop=True)
    return {"points": df.to_dict(orient="records")}
