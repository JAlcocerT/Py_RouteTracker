from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.state import video_store
from app.laps.detection import detect_laps, get_coordinates_at_time
from app.laps.extrema import compare_laps

router = APIRouter(prefix="/api/videos", tags=["laps"])


class LapDetectRequest(BaseModel):
    start_time_s: float
    radius_m: float = 15.0
    min_lap_time_s: float = 30.0


def _load_telemetry(video_id: str) -> pd.DataFrame:
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    path = video_store.telemetry_path(video_id)
    if not meta.telemetry_ready or not path.exists():
        raise HTTPException(425, "telemetry extraction is not finished yet")
    return pd.read_parquet(path)


@router.post("/{video_id}/laps/detect")
async def detect_laps_endpoint(video_id: str, body: LapDetectRequest):
    df = _load_telemetry(video_id)

    start_lat, start_lon = get_coordinates_at_time(df, body.start_time_s)
    result = detect_laps(df, start_lat, start_lon, radius_m=body.radius_m, min_lap_time_s=body.min_lap_time_s)

    result.annotated_df.to_parquet(video_store.laps_annotated_path(video_id))
    result.lap_table.to_parquet(video_store.laps_table_path(video_id))
    video_store.save_lap_indices(video_id, result.lap_indices)
    video_store.update(video_id, laps_ready=True, lap_start_time_s=body.start_time_s)

    return {
        "lap_count": max(0, len(result.lap_indices) - 1),
        "lap_table": result.lap_table.to_dict(orient="records"),
        "start_lat": start_lat,
        "start_lon": start_lon,
    }


@router.get("/{video_id}/laps")
async def get_laps(video_id: str):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    if not meta.laps_ready:
        raise HTTPException(404, "laps have not been detected yet for this video")
    lap_table = pd.read_parquet(video_store.laps_table_path(video_id))
    return {"lap_table": lap_table.to_dict(orient="records")}


@router.get("/{video_id}/laps/compare")
async def compare_laps_endpoint(video_id: str, lap_a: int, lap_b: int, extrema_window: int = 30):
    meta = video_store.get(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    if not meta.laps_ready:
        raise HTTPException(404, "laps have not been detected yet for this video")

    df = pd.read_parquet(video_store.laps_annotated_path(video_id))
    lap_table = pd.read_parquet(video_store.laps_table_path(video_id))
    lap_indices = video_store.load_lap_indices(video_id)

    try:
        comparison = compare_laps(df, lap_table, lap_indices, lap_a, lap_b, extrema_window=extrema_window)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "lap_a": comparison.lap_a,
        "lap_b": comparison.lap_b,
        "duration_a": comparison.duration_a,
        "duration_b": comparison.duration_b,
        "series_a": comparison.series_a.to_dict(orient="records"),
        "series_b": comparison.series_b.to_dict(orient="records"),
        "maxima_a": comparison.maxima_a,
        "minima_a": comparison.minima_a,
        "maxima_b": comparison.maxima_b,
        "minima_b": comparison.minima_b,
    }
