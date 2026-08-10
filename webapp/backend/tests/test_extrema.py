import numpy as np
import pandas as pd
import pytest

from app.laps.extrema import compare_laps, find_local_extrema


def test_find_local_extrema_single_peak():
    values = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    maxima = find_local_extrema(values, window=3, mode="max")
    assert maxima == [(5, 5)]


def test_find_local_extrema_single_trough():
    values = [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    minima = find_local_extrema(values, window=3, mode="min")
    assert minima == [(5, 0)]


def test_find_local_extrema_empty_when_window_too_wide():
    values = [1, 2, 3]
    assert find_local_extrema(values, window=5, mode="max") == []


def _two_lap_fixture():
    # two laps: lap 1 has a lower peak speed than lap 2, everything else equal
    t = np.arange(0, 40, 0.5)
    lap1_speed = 100 + 20 * np.sin(2 * np.pi * t / 40)
    lap2_speed = 100 + 30 * np.sin(2 * np.pi * t / 40)

    df = pd.DataFrame({
        "time": np.concatenate([t, t + 40]),
        "lat": 0.0,
        "lon": 0.0,
        "speed": np.concatenate([lap1_speed, lap2_speed]),
        "lat_g": 0.0,
        "lon_g": 0.0,
    })
    lap_indices = [0, len(t), len(t) * 2 - 1]
    lap_table = pd.DataFrame([
        {"lap": 1, "start_time": 0.0, "end_time": 40.0, "duration": 40.0, "avg_speed": 100.0, "max_speed": 120.0},
        {"lap": 2, "start_time": 40.0, "end_time": 79.5, "duration": 39.5, "avg_speed": 100.0, "max_speed": 130.0},
    ])
    return df, lap_table, lap_indices


def test_compare_laps_returns_expected_shape():
    df, lap_table, lap_indices = _two_lap_fixture()
    comparison = compare_laps(df, lap_table, lap_indices, lap_a=1, lap_b=2, extrema_window=5)

    assert comparison.lap_a == 1
    assert comparison.lap_b == 2
    assert comparison.duration_a == pytest.approx(40.0)
    assert comparison.duration_b == pytest.approx(39.5)
    assert not comparison.series_a.empty
    assert not comparison.series_b.empty
    assert len(comparison.maxima_b) >= 1
    # lap 2 has a higher-amplitude sine wave -> higher peak speed
    assert max(v for _, v in comparison.maxima_b) > max(v for _, v in comparison.maxima_a)


def test_compare_laps_rejects_out_of_range_lap_numbers():
    df, lap_table, lap_indices = _two_lap_fixture()
    with pytest.raises(ValueError):
        compare_laps(df, lap_table, lap_indices, lap_a=1, lap_b=5)
