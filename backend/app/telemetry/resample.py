"""Shared time-grid resampling, used by every telemetry source.

Different sources sample at different native rates (GoPro GPS ~18Hz, GPMD
accel ~200Hz, a phone/Garmin GPX track often 1Hz) — everything downstream
(lap detection, rendering) wants one uniform per-frame time grid instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_speed_outliers(speed: pd.Series, window: int = 3) -> pd.Series:
    """Rejects isolated GPS speed spikes with a rolling median filter.

    Both telemetry sources are exposed to this: a GPS chip's own reported
    speed can glitch during multipath/low-satellite-count, and a
    position-delta-derived speed (external_gpx) is even more sensitive --
    a couple of meters of ordinary position jitter divided by a small time
    delta between fixes produces a wildly implausible instantaneous speed.
    A short centered median only rejects a sample that disagrees with both
    its neighbors, which a genuine (sustained) acceleration/braking change
    doesn't -- it doesn't get flattened, only single-sample noise does.
    """
    if len(speed) < window:
        return speed
    return speed.rolling(window, center=True, min_periods=1).median()


def resample_to_grid(df: pd.DataFrame, duration_sec: float, target_fps: float, value_cols: list[str]) -> pd.DataFrame:
    """Interpolates `value_cols` (indexed by df['time']) onto a uniform grid.

    Returns a DataFrame with a 'time' column plus interpolated `value_cols`,
    one row per 1/target_fps step from 0 to duration_sec.
    """
    if df.empty:
        return pd.DataFrame(columns=["time", *value_cols])

    t_target = np.arange(0, duration_sec, 1 / target_fps)
    indexed = df.set_index("time")[value_cols]
    resampled = (
        indexed.reindex(indexed.index.union(t_target))
        .interpolate(method="index")
        # interpolate(method="index") only fills *between* known points --
        # grid steps before the first sample or after the last (e.g. a video
        # trimmed slightly longer than the telemetry's own coverage) are left
        # NaN, which breaks both downstream math and JSON serialization.
        .ffill()
        .bfill()
        .reindex(t_target)
        .reset_index()
        .rename(columns={"index": "time"})
    )
    return resampled
