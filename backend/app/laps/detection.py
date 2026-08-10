"""Lap detection: ported from legacy/overlay/lap_timer_v7.py:94-144 and
legacy/overlay/racing_hud_v7.py:195-249 (the two scripts had nearly identical
copies of this logic; this is the merged, parameterized version — the
`racing_hud_v7` variant additionally annotated every row with its lap
number and the last completed lap time, which the HUD renderer needs, so
that's kept as the default behavior).

Detection works by finding the closest approach to a start/finish
coordinate within `radius_m`, at least `min_lap_time_s` apart — not a
simple "crossed into the circle" trigger, which would double-count noisy
GPS points near the line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000


@dataclass
class LapDetectionResult:
    annotated_df: pd.DataFrame
    lap_indices: list[int]
    lap_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters. Works with scalars or numpy arrays."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.subtract(lat2, lat1))
    dlambda = np.radians(np.subtract(lon2, lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_coordinates_at_time(df: pd.DataFrame, target_time: float) -> tuple[float, float]:
    """Finds the (lat, lon) of the sample closest to `target_time` seconds —
    used to let the user pick a lap start/finish line by pointing at a time
    in the video instead of typing raw coordinates."""
    idx = (df["time"] - target_time).abs().idxmin()
    row = df.loc[idx]
    return float(row["lat"]), float(row["lon"])


def detect_laps(
    df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    radius_m: float = 15.0,
    min_lap_time_s: float = 30.0,
) -> LapDetectionResult:
    if df.empty:
        return LapDetectionResult(annotated_df=df.copy(), lap_indices=[], lap_table=pd.DataFrame())

    lap_indices: list[int] = []
    last_crossing_time = -min_lap_time_s
    in_zone = False
    best_dist = float("inf")
    best_idx = -1

    dist_to_start = haversine_distance_m(df["lat"].to_numpy(), df["lon"].to_numpy(), start_lat, start_lon)

    for i, (row_time, dist) in enumerate(zip(df["time"].to_numpy(), dist_to_start)):
        if (row_time - last_crossing_time) > min_lap_time_s:
            if dist < radius_m:
                in_zone = True
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            elif in_zone:
                lap_indices.append(best_idx)
                last_crossing_time = df["time"].iloc[best_idx]
                in_zone = False
                best_dist = float("inf")

    lap_table_rows = []
    for k in range(1, len(lap_indices)):
        s_idx, e_idx = lap_indices[k - 1], lap_indices[k]
        lap_slice = df.iloc[s_idx:e_idx]
        lap_table_rows.append({
            "lap": k,
            "start_time": df.iloc[s_idx]["time"],
            "end_time": df.iloc[e_idx]["time"],
            "duration": df.iloc[e_idx]["time"] - df.iloc[s_idx]["time"],
            "avg_speed": lap_slice["speed"].mean(),
            "max_speed": lap_slice["speed"].max(),
        })
    lap_table = pd.DataFrame(lap_table_rows)

    annotated = df.copy()
    annotated["lap"] = 0
    annotated["last_lap_s"] = 0.0
    current_lap = 1
    prev_idx = 0
    for idx in lap_indices:
        annotated.loc[prev_idx:idx, "lap"] = current_lap
        if prev_idx > 0:
            annotated.loc[idx:, "last_lap_s"] = annotated.iloc[idx]["time"] - annotated.iloc[prev_idx]["time"]
        prev_idx = idx
        current_lap += 1
    annotated.loc[prev_idx:, "lap"] = current_lap

    return LapDetectionResult(annotated_df=annotated, lap_indices=lap_indices, lap_table=lap_table)
