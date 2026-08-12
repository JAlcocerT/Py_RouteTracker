"""Regression coverage for a real bug report: a go-kart session (genuine
max speed ~85 km/h) showed a lap `max_speed` of ~300 km/h. Root cause was
that neither telemetry source did any outlier rejection on `speed` --
`external_gpx.py` derives it from position deltas, which is exactly the
kind of thing a couple of meters of ordinary GPS jitter divided by a small
time delta between fixes can blow up into an implausible instantaneous
speed; a GPS chip's own reported speed (gopro_embedded.py) can glitch the
same way during multipath/low-satellite-count. See
`app.telemetry.resample.smooth_speed_outliers`.
"""

import numpy as np
import pandas as pd
import pytest

from app.laps.detection import detect_laps, get_coordinates_at_time
from app.telemetry.resample import smooth_speed_outliers
from tests.test_laps_detection import _circular_track_df


def test_smooth_speed_outliers_rejects_isolated_spike():
    speed = pd.Series([84.0, 85.0, 86.0, 300.0, 85.0, 84.0, 85.0])
    cleaned = smooth_speed_outliers(speed)
    assert cleaned.max() < 100
    # genuine neighboring values are left essentially untouched
    assert cleaned.iloc[0] == pytest.approx(84.5, abs=1.0)


def test_smooth_speed_outliers_preserves_sustained_change():
    # a real acceleration lasting several samples must not be flattened --
    # only a single-sample outlier should be rejected
    speed = pd.Series([20.0, 20.0, 60.0, 65.0, 70.0, 70.0, 70.0])
    cleaned = smooth_speed_outliers(speed)
    assert cleaned.iloc[-1] == pytest.approx(70.0)
    assert cleaned.iloc[3] > 50  # mid-acceleration sample not smoothed away


def test_smooth_speed_outliers_short_series_is_a_noop():
    speed = pd.Series([10.0, 20.0])
    assert smooth_speed_outliers(speed).tolist() == [10.0, 20.0]


def test_smooth_speed_outliers_window_scales_with_sample_rate():
    """Regression test for the audit's #2 finding: a flat window=5 only
    outvotes a burst of up to ~2 bad samples. A real GPS signal-loss burst
    (a bridge, a tunnel, dense tree cover) at a GoPro-like ~20Hz native rate
    routinely spans more samples than that -- passing `time` should widen
    the window to actually cover it, where the flat default can't.
    """
    n = 30
    baseline = 85.0
    speed = pd.Series([baseline] * n, dtype=float)
    burst_start, burst_len = 13, 4
    speed.iloc[burst_start:burst_start + burst_len] = 300.0

    time = np.arange(n) * 0.05  # 20 Hz

    without_time = smooth_speed_outliers(speed.copy())
    assert without_time.max() > 150  # flat window=5 can't outvote this burst

    with_time = smooth_speed_outliers(speed.copy(), time=time)
    assert with_time.max() < 100  # rate-derived window (~10 samples) can


def test_smooth_speed_outliers_rejects_two_sample_burst():
    # A brief multipath dropout (e.g. passing under a bridge/netting on a
    # go-kart track) commonly glitches more than one consecutive GPS fix,
    # not just one -- a plain window=3 median can't outvote that (the bad
    # samples dominate their own window), so this must actually be rejected.
    speed = pd.Series([84.0, 85.0, 86.0, 300.0, 305.0, 85.0, 84.0, 85.0])
    cleaned = smooth_speed_outliers(speed)
    assert cleaned.max() < 100


def test_detect_laps_max_speed_ignores_single_sample_gps_spike():
    # 85 km/h go-kart lap with one glitched sample injected mid-lap
    df = _circular_track_df(n_laps=2, lap_time_s=40, dt=0.5, track_radius_m=150.31)
    assert df["speed"].iloc[0] == pytest.approx(85.0, abs=1.0)

    spike_idx = len(df) // 4
    df.loc[spike_idx, "speed"] = 300.0
    df["speed"] = smooth_speed_outliers(df["speed"])

    start_lat, start_lon = get_coordinates_at_time(df, target_time=0.0)
    result = detect_laps(df, start_lat, start_lon, radius_m=15.0, min_lap_time_s=20.0)

    assert len(result.lap_table) >= 1
    assert result.lap_table["max_speed"].max() < 100


def test_detect_laps_max_speed_ignores_two_sample_gps_burst():
    # same as above, but with a two-consecutive-sample glitch burst, which
    # is the more realistic failure mode and the one the original
    # window=3-median fix didn't actually cover
    df = _circular_track_df(n_laps=2, lap_time_s=40, dt=0.5, track_radius_m=150.31)
    spike_idx = len(df) // 4
    df.loc[spike_idx, "speed"] = 300.0
    df.loc[spike_idx + 1, "speed"] = 305.0
    df["speed"] = smooth_speed_outliers(df["speed"])

    start_lat, start_lon = get_coordinates_at_time(df, target_time=0.0)
    result = detect_laps(df, start_lat, start_lon, radius_m=15.0, min_lap_time_s=20.0)

    assert len(result.lap_table) >= 1
    assert result.lap_table["max_speed"].max() < 100
