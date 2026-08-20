from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.ffmpeg_utils import get_video_duration, get_video_fps
from app.core.state import job_manager, video_store
from app.core.video_join import IncompatiblePartsError, join_videos, validate_join_compatible
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


def _register_and_extract(
    video_id: str,
    video_path: Path,
    filename: str,
    source_type: str,
    gpx_path: Optional[Path],
    video_start_time: Optional[str],
    offset_sec: float,
    target_fps: Optional[float],
) -> dict:
    """Shared tail of both upload_video and join_and_upload_video: record
    the (now on-disk, single-file) video's metadata and kick off telemetry
    extraction. A joined video is, from this point on, indistinguishable
    from a single-file upload -- the join happens entirely before this.
    """
    try:
        duration_sec = get_video_duration(video_path)
    except Exception as exc:
        raise HTTPException(500, f"could not read video duration: {exc}") from exc

    if target_fps is None:
        # Default to the footage's own frame rate rather than a flat
        # constant -- HUD frames are drawn one per telemetry row and
        # composited 1:1 onto the trimmed clip, so matching the real rate
        # means every video frame gets its own HUD frame instead of relying
        # on ffmpeg's overlay filter to duplicate/drop frames to reconcile a
        # mismatch. Falls back to the configured constant if ffprobe can't
        # read it (e.g. an unusual container).
        try:
            target_fps = get_video_fps(video_path)
        except Exception:
            target_fps = settings.default_target_fps

    meta = VideoMeta(
        id=video_id,
        filename=filename,
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


@router.post("")
async def upload_video(
    video: UploadFile,
    source_type: str = Form(...),
    gpx: Optional[UploadFile] = None,
    video_start_time: Optional[str] = Form(None),
    offset_sec: float = Form(0.0),
    target_fps: Optional[float] = Form(None),
):
    if source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of {VALID_SOURCE_TYPES}")
    if source_type == "external_gpx" and gpx is None:
        raise HTTPException(400, "external_gpx source_type requires a gpx file upload")

    video_id = video_store.new_id()
    video_dir = video_store.video_dir(video_id)

    video_path = _save_upload(video, video_dir / video.filename)
    gpx_path = _save_upload(gpx, video_dir / gpx.filename) if gpx is not None else None

    return _register_and_extract(
        video_id, video_path, video.filename, source_type, gpx_path,
        video_start_time, offset_sec, target_fps,
    )


@router.post("/join")
async def join_and_upload_video(
    videos: list[UploadFile] = File(...),
    source_type: str = Form(...),
    gpx: Optional[UploadFile] = None,
    video_start_time: Optional[str] = Form(None),
    offset_sec: float = Form(0.0),
    target_fps: Optional[float] = Form(None),
):
    """Joins multiple split parts of one continuous recording (e.g. a
    chaptered GoPro clip) into a single lossless file before running the
    normal upload/extraction pipeline on it -- see app.core.video_join for
    why this preserves embedded GPS/telemetry that a naive re-encode join
    would drop. `videos` must already be in the correct playback order;
    the frontend is responsible for arranging/letting the user reorder
    them before submitting.
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of {VALID_SOURCE_TYPES}")
    if source_type == "external_gpx" and gpx is None:
        raise HTTPException(400, "external_gpx source_type requires a gpx file upload")
    if len(videos) < 2:
        raise HTTPException(400, "provide at least two video parts to join")

    video_id = video_store.new_id()
    video_dir = video_store.video_dir(video_id)
    parts_dir = video_dir / "parts"

    part_paths = [_save_upload(v, parts_dir / v.filename) for v in videos]
    gpx_path = _save_upload(gpx, video_dir / gpx.filename) if gpx is not None else None

    try:
        validate_join_compatible(part_paths)
    except IncompatiblePartsError as exc:
        shutil.rmtree(parts_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc

    joined_name = f"joined_{Path(videos[0].filename).stem}.mp4"
    joined_path = video_dir / joined_name
    try:
        join_videos(part_paths, joined_path)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"failed to join video parts: {exc}") from exc
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)

    return _register_and_extract(
        video_id, joined_path, joined_name, source_type, gpx_path,
        video_start_time, offset_sec, target_fps,
    )


@router.get("/{video_id}")
async def get_video(video_id: str):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    data = meta.to_json()
    created = datetime.fromisoformat(meta.created_at)
    data["expires_at"] = (created + timedelta(minutes=settings.retention_minutes)).isoformat()
    return data


@router.get("/{video_id}/source")
async def get_video_source(video_id: str):
    """Streams back the stored video file as-is (the joined output, for a
    /join upload). Naively concatenating raw MP4 bytes client-side doesn't
    produce a playable file, so a joined video's frontend preview/trim UI
    fetches it back from here once instead -- everything downstream of
    that (trim, render) already works from the server-side path and never
    needed this."""
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    path = Path(meta.video_path)
    if not path.exists():
        raise HTTPException(404, "video file not found (it may have expired)")
    return FileResponse(path, media_type="video/mp4", filename=meta.filename)


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
