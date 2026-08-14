"""Lap detection: ported from legacy/overlay/lap_timer_v7.py:94-144 and
legacy/overlay/racing_hud_v7.py:195-249 (the two scripts had nearly identical
copies of this logic; this is the merged, parameterized version — the
`racing_hud_v7` variant additionally annotated every row with its lap
number and the last completed lap time, which the HUD renderer needs, so
that's kept as the default behavior).

Detection works within a start/finish zone: a circle of `radius_m` around
`(start_lat, start_lon)`, entered and left at least `min_lap_time_s` apart —
not a simple "crossed into the circle" trigger, which would double-count
noisy GPS points near the line.

The crossing *instant* within that zone is a virtual gate line through the
start point, perpendicular to the direction of travel (estimated from the
GPS track itself the first time a zone is entered, then reused for every
later crossing -- a closed circuit is driven the same direction every lap),
interpolated to sub-sample precision between whichever two samples straddle
it -- not just "whichever GPS sample happened to pass closest to the start
point," which is a different question. On a real lap, at typical racing
speeds, that gap between "closest sample" and "actual line crossing" is
easily hundreds of milliseconds: the closest-approach point is wherever the
kart's arc happens to nearest the coordinate, which only coincides with the
real crossing when the kart crosses the line exactly radially through the
center of the zone -- rare in practice, especially when the start/finish is
near a corner rather than a straight. Falls back to the old nearest-sample
behavior if no clean sign-change is found (e.g. a heading estimate thrown
off by GPS noise, or a session too short to establish one).
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


def _local_meters(lat: np.ndarray, lon: np.ndarray, origin_lat: float, origin_lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular (east_m, north_m) of each point relative to an origin
    -- a flat-plane approximation, fine at track scale (hundreds of meters
    at most), needed because degrees of latitude and longitude aren't the
    same size in meters (longitude shrinks by cos(latitude))."""
    origin_lat_rad = np.radians(origin_lat)
    east_m = np.radians(lon - origin_lon) * EARTH_RADIUS_M * np.cos(origin_lat_rad)
    north_m = np.radians(lat - origin_lat) * EARTH_RADIUS_M
    return east_m, north_m


def _estimate_heading(east_m: np.ndarray, north_m: np.ndarray, idx: int, window: int) -> tuple[float, float] | None:
    """Unit (east, north) direction of travel around sample `idx`, from the
    net displacement over +/- `window` samples -- averaging out GPS jitter
    that a single-sample finite difference would be very sensitive to.
    None if the vehicle wasn't actually moving there (can't establish a
    direction from zero displacement)."""
    lo, hi = max(0, idx - window), min(len(east_m) - 1, idx + window)
    dx, dy = east_m[hi] - east_m[lo], north_m[hi] - north_m[lo]
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return None
    return dx / norm, dy / norm


def _resolve_crossing(
    east_m: np.ndarray,
    north_m: np.ndarray,
    t: np.ndarray,
    zone_start_i: int,
    zone_end_i: int,
    best_idx: int,
    heading: tuple[float, float] | None,
) -> tuple[int, float]:
    """Returns (row index, interpolated time) of the start/finish crossing.

    With a known heading, the crossing is where the signed projection of
    position (relative to the start point, which is the origin `east_m`/
    `north_m` are already measured from) onto that heading direction passes
    through zero -- i.e. the moment the vehicle is neither behind nor ahead
    of the start point along its direction of travel, found by linear
    interpolation between whichever two samples straddle that sign change.
    Falls back to `best_idx` (the old nearest-approach sample) if no sign
    change turns up in the zone -- a heading estimate that doesn't actually
    describe this crossing's geometry shouldn't invent a worse answer than
    the simple fallback."""
    if heading is not None:
        hx, hy = heading
        lo, hi = max(0, zone_start_i - 1), min(len(t) - 1, zone_end_i)
        proj = east_m[lo:hi + 1] * hx + north_m[lo:hi + 1] * hy
        for k in range(len(proj) - 1):
            p0, p1 = proj[k], proj[k + 1]
            if (p0 <= 0) != (p1 <= 0):
                i0, i1 = lo + k, lo + k + 1
                frac = 0.0 if p1 == p0 else min(max(-p0 / (p1 - p0), 0.0), 1.0)
                cross_t = t[i0] + frac * (t[i1] - t[i0])
                return (i0 if frac < 0.5 else i1), float(cross_t)
    return best_idx, float(t[best_idx])


def detect_laps(
    df: pd.DataFrame,
    start_lat: float,
    start_lon: float,
    radius_m: float = 15.0,
    min_lap_time_s: float = 30.0,
) -> LapDetectionResult:
    if df.empty:
        return LapDetectionResult(annotated_df=df.copy(), lap_indices=[], lap_table=pd.DataFrame())

    t = df["time"].to_numpy()
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    dist_to_start = haversine_distance_m(lat, lon, start_lat, start_lon)
    east_m, north_m = _local_meters(lat, lon, start_lat, start_lon)

    median_dt = np.median(np.diff(t)) if len(t) >= 2 else 0.0
    heading_window = max(3, round(0.5 / median_dt)) if median_dt > 0 else 3
    heading: tuple[float, float] | None = None

    lap_indices: list[int] = []
    crossing_times: list[float] = []
    last_crossing_time = -min_lap_time_s
    in_zone = False
    best_dist = float("inf")
    best_idx = -1
    zone_start_i = -1

    for i in range(len(df)):
        row_time, dist = t[i], dist_to_start[i]
        if (row_time - last_crossing_time) > min_lap_time_s:
            if dist < radius_m:
                if not in_zone:
                    zone_start_i = i
                in_zone = True
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            elif in_zone:
                cross_idx, cross_t = _resolve_crossing(east_m, north_m, t, zone_start_i, i, best_idx, heading)
                if heading is None:
                    heading = _estimate_heading(east_m, north_m, best_idx, heading_window)
                lap_indices.append(cross_idx)
                crossing_times.append(cross_t)
                last_crossing_time = cross_t
                in_zone = False
                best_dist = float("inf")

    lap_table_rows = []
    for k in range(1, len(lap_indices)):
        s_idx, e_idx = lap_indices[k - 1], lap_indices[k]
        lap_slice = df.iloc[s_idx:e_idx]
        lap_table_rows.append({
            "lap": k,
            "start_time": crossing_times[k - 1],
            "end_time": crossing_times[k],
            "duration": crossing_times[k] - crossing_times[k - 1],
            "avg_speed": lap_slice["speed"].mean(),
            "max_speed": lap_slice["speed"].max(),
        })
    lap_table = pd.DataFrame(lap_table_rows)

    annotated = df.copy()
    annotated["lap"] = 0
    annotated["last_lap_s"] = 0.0
    current_lap = 1
    prev_idx = 0
    for k, idx in enumerate(lap_indices):
        annotated.loc[prev_idx:idx, "lap"] = current_lap
        if prev_idx > 0:
            annotated.loc[idx:, "last_lap_s"] = crossing_times[k] - crossing_times[k - 1]
        prev_idx = idx
        current_lap += 1
    annotated.loc[prev_idx:, "lap"] = current_lap

    # Live "time since the current lap's own start crossing" for every row
    # (the dynamic in-HUD chronometer) -- computed independently of the
    # `lap`/`last_lap_s` loop above rather than folded into it, since that
    # loop's boundary row (a crossing sample) gets written twice, by both
    # the segment it closes and the one it opens, and whichever write
    # happens last wins; a vectorized "most recent crossing at or before
    # this row's own timestamp" search sidesteps that ambiguity and, unlike
    # `lap_indices` (raw sample positions), keys off the interpolated
    # crossing times themselves for frame-accurate results.
    crossing_arr = np.asarray(crossing_times, dtype=float)
    row_times = annotated["time"].to_numpy()
    if len(crossing_arr):
        pos = np.searchsorted(crossing_arr, row_times, side="right")
        segment_start = np.where(pos > 0, crossing_arr[np.clip(pos - 1, 0, len(crossing_arr) - 1)], 0.0)
    else:
        segment_start = np.zeros_like(row_times)
    annotated["lap_elapsed_s"] = row_times - segment_start

    return LapDetectionResult(annotated_df=annotated, lap_indices=lap_indices, lap_table=lap_table)
