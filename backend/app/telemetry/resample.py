"""Shared time-grid resampling, used by every telemetry source.

Different sources sample at different native rates (GoPro GPS ~18Hz, GPMD
accel ~200Hz, a phone/Garmin GPX track often 1Hz) — everything downstream
(lap detection, rendering) wants one uniform per-frame time grid instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_speed_outliers(speed: pd.Series, window: int = 5, mad_threshold: float = 5.0) -> pd.Series:
    """Rejects implausible GPS speed spikes with a Hampel filter (rolling
    median + median-absolute-deviation threshold).

    Both telemetry sources are exposed to this: a GPS chip's own reported
    speed can glitch during multipath/low-satellite-count, and a
    position-delta-derived speed (external_gpx) is even more sensitive --
    a couple of meters of ordinary position jitter divided by a small time
    delta between fixes produces a wildly implausible instantaneous speed.

    A plain small rolling median (the original approach here) only rejects
    a sample that disagrees with *both* its immediate neighbors -- a burst
    of two or more consecutive bad fixes (e.g. a trackside structure
    blocking sky view for a fraction of a second, common right where a
    go-kart track passes under a bridge/netting) drags the window's median
    along with it and survives untouched. Using a wider window plus an
    explicit outlier test (deviation from the local median, scaled against
    the local median absolute deviation) instead of blanket-replacing every
    sample lets the window stay wide enough to outvote a short burst while
    only actually touching samples that are statistically implausible, so a
    genuine (sustained) acceleration/braking ramp is left alone. This is
    still fundamentally limited by "majority of the window must be good
    fixes" -- a dropout lasting more than about half the window can't be
    distinguished from a real sustained speed change by this alone.
    """
    if len(speed) < window:
        return speed
    median = speed.rolling(window, center=True, min_periods=1).median()
    deviation = (speed - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=1).median()
    # 1.4826 makes MAD comparable to a standard deviation for normally
    # distributed data; floored so a near-constant local window (MAD ~ 0)
    # doesn't make ordinary tiny wobble register as an outlier.
    scaled_mad = (mad * 1.4826).clip(lower=1.0)
    is_outlier = deviation > mad_threshold * scaled_mad
    return speed.where(~is_outlier, median)


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
