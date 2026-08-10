"""Common contract for telemetry sources.

Every source (GoPro-embedded metadata, an external GPX file, future FIT
support, ...) produces the same tidy DataFrame so the rest of the pipeline
(lap detection, rendering) never needs to know where the data came from.

Columns:
    time    float seconds, 0-based, relative to the start of the video
    lat     float degrees
    lon     float degrees
    speed   float km/h
    lat_g   float lateral G-force (0.0 if the source has no accelerometer data)
    lon_g   float longitudinal G-force (0.0 if unavailable)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

TELEMETRY_COLUMNS = ["time", "lat", "lon", "speed", "lat_g", "lon_g"]


@dataclass
class TelemetryResult:
    df: pd.DataFrame
    source_name: str
    has_accel: bool

    def __post_init__(self) -> None:
        missing = [c for c in TELEMETRY_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"TelemetryResult missing required columns: {missing}")


class TelemetrySource(Protocol):
    """A pluggable telemetry provider for a given (video, [aux file]) pair."""

    def extract(self, video_path: Path, duration_sec: float) -> TelemetryResult: ...


def empty_result(source_name: str) -> TelemetryResult:
    return TelemetryResult(
        df=pd.DataFrame(columns=TELEMETRY_COLUMNS),
        source_name=source_name,
        has_accel=False,
    )
