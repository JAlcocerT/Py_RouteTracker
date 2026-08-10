"""Telemetry source for cameras with no embedded GPS track.

Many action cams (DJI, Insta360, older GoPros) and most external GPS
trackers (phone apps, Garmin, Polar) don't embed GPMD in the video at all —
the user has a separate .gpx file instead. This source aligns that GPX
track onto the video's own timeline using either an explicit offset or the
video's real-world start time (matched against GPX point timestamps).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd

from app.laps.detection import haversine_distance_m
from app.telemetry.resample import resample_to_grid
from app.telemetry.sources.base import TelemetryResult, empty_result


def load_gpx_points(gpx_path: Path) -> pd.DataFrame:
    """Flattens every track/segment point in a GPX file into one DataFrame."""
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append({
                    "timestamp": point.time,
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "ele": point.elevation,
                })

    df = pd.DataFrame(rows)
    if df.empty or df["timestamp"].isna().any():
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_speed_kmh(df: pd.DataFrame) -> pd.Series:
    """Derives speed from consecutive GPX points (most consumer GPX files
    don't carry a <speed> extension, so this is computed, not read)."""
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    seconds = df["timestamp"].apply(lambda t: t.timestamp()).to_numpy()

    dist_m = np.zeros(len(df))
    dist_m[1:] = haversine_distance_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(seconds, prepend=seconds[0] if len(seconds) else 0)
    dt[dt <= 0] = np.nan

    speed_ms = dist_m / dt
    speed_ms = np.nan_to_num(speed_ms, nan=0.0)
    return pd.Series(speed_ms * 3.6, index=df.index)


class ExternalGpxSource:
    """TelemetrySource backed by a standalone GPX file, time-aligned to the video."""

    def __init__(self, gpx_path: Path, target_fps: float = 30.0, offset_sec: float = 0.0, video_start_time: datetime | None = None):
        self.gpx_path = Path(gpx_path)
        self.target_fps = target_fps
        self.offset_sec = offset_sec
        self.video_start_time = video_start_time

    def extract(self, video_path: Path, duration_sec: float) -> TelemetryResult:
        points = load_gpx_points(self.gpx_path)
        if points.empty:
            return empty_result("external_gpx")

        points["speed"] = compute_speed_kmh(points)

        if self.video_start_time is not None:
            points["time"] = (points["timestamp"] - self.video_start_time).dt.total_seconds()
        else:
            first_ts = points["timestamp"].iloc[0]
            points["time"] = (points["timestamp"] - first_ts).dt.total_seconds() + self.offset_sec

        windowed = points[(points["time"] >= 0) & (points["time"] <= duration_sec)]
        if len(windowed) < 2:
            return empty_result("external_gpx")

        resampled = resample_to_grid(windowed, duration_sec, self.target_fps, ["lat", "lon", "speed"])
        resampled["lat_g"] = 0.0
        resampled["lon_g"] = 0.0
        return TelemetryResult(df=resampled, source_name="external_gpx", has_accel=False)
