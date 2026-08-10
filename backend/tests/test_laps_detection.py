import numpy as np
import pandas as pd
import pytest

from app.laps.detection import detect_laps, get_coordinates_at_time, haversine_distance_m

DEG_PER_METER_AT_EQUATOR = 1 / 111_000


def _circular_track_df(n_laps: int, lap_time_s: float, dt: float, track_radius_m: float = 100.0) -> pd.DataFrame:
    """Synthesizes GPS samples for a vehicle looping a circular track centered
    on the equator (so 1 degree of lat/lon ~= 111km, no cos(lat) correction
    needed) — enough to exercise detect_laps without needing a real GPX/exif
    fixture with a known start/finish line."""
    radius_deg = track_radius_m * DEG_PER_METER_AT_EQUATOR
    t = np.arange(0, n_laps * lap_time_s, dt)
    theta = 2 * np.pi * (t % lap_time_s) / lap_time_s

    lat = radius_deg * np.cos(theta)
    lon = radius_deg * np.sin(theta)

    # constant angular speed -> constant linear speed
    circumference_m = 2 * np.pi * track_radius_m
    speed_kmh = (circumference_m / lap_time_s) * 3.6

    return pd.DataFrame({
        "time": t,
        "lat": lat,
        "lon": lon,
        "speed": np.full_like(t, speed_kmh),
        "lat_g": 0.0,
        "lon_g": 0.0,
    })


def test_haversine_known_distance():
    # 1 degree of latitude is ~111.19 km
    d = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111_195, rel=1e-3)


def test_haversine_vectorized_matches_scalar():
    lat1 = np.array([0.0, 10.0])
    lon1 = np.array([0.0, 20.0])
    scalar_results = [haversine_distance_m(a, b, 0.0, 0.0) for a, b in zip(lat1, lon1)]
    vector_results = haversine_distance_m(lat1, lon1, 0.0, 0.0)
    np.testing.assert_allclose(vector_results, scalar_results)


def test_get_coordinates_at_time():
    df = _circular_track_df(n_laps=1, lap_time_s=60, dt=0.5)
    lat, lon = get_coordinates_at_time(df, target_time=0.0)
    assert lat == pytest.approx(100 * DEG_PER_METER_AT_EQUATOR)
    assert lon == pytest.approx(0.0, abs=1e-9)


def test_detect_laps_finds_expected_lap_count():
    df = _circular_track_df(n_laps=4, lap_time_s=60, dt=0.5, track_radius_m=100.0)
    start_lat, start_lon = get_coordinates_at_time(df, target_time=0.0)

    result = detect_laps(df, start_lat, start_lon, radius_m=15.0, min_lap_time_s=30.0)

    # detect_laps requires (time - last_crossing) > min_lap_time_s (strict),
    # so the very first sample at t=0 (0 - (-min_lap_time_s) == min_lap_time_s)
    # never qualifies as a crossing candidate -- this is inherited unchanged
    # from overlay/lap_timer_v7.py's own boundary condition. The first
    # crossing is therefore registered a half-step later, at t=0.5s, giving
    # 4 crossings (~0.5s, 60s, 120s, 180s) and 3 completed lap rows over
    # this 240s / 4-lap synthetic dataset.
    assert len(result.lap_indices) == 4
    assert len(result.lap_table) == 3
    assert result.lap_table["duration"].apply(lambda d: d == pytest.approx(60.0, abs=1.0)).all()


def test_detect_laps_annotates_lap_numbers_monotonically():
    df = _circular_track_df(n_laps=3, lap_time_s=40, dt=0.5, track_radius_m=80.0)
    start_lat, start_lon = get_coordinates_at_time(df, target_time=0.0)

    result = detect_laps(df, start_lat, start_lon, radius_m=15.0, min_lap_time_s=20.0)

    assert "lap" in result.annotated_df.columns
    assert result.annotated_df["lap"].is_monotonic_increasing
    # the trailing segment after the last crossing is labeled one past the
    # number of crossings (an uncompleted lap in progress) -- see detect_laps
    assert result.annotated_df["lap"].max() == len(result.lap_indices) + 1


def test_detect_laps_empty_df_returns_empty_result():
    result = detect_laps(pd.DataFrame(columns=["time", "lat", "lon", "speed"]), 0.0, 0.0)
    assert result.lap_indices == []
    assert result.lap_table.empty


def test_detect_laps_respects_min_lap_time_ignores_noise_near_line():
    # a vehicle that stops right on the start line for a while shouldn't
    # register dozens of "laps" from GPS jitter
    n = 200
    df = pd.DataFrame({
        "time": np.arange(n) * 0.5,
        "lat": np.zeros(n) + np.random.default_rng(0).normal(0, 1e-7, n),
        "lon": np.zeros(n),
        "speed": np.zeros(n),
        "lat_g": 0.0,
        "lon_g": 0.0,
    })
    result = detect_laps(df, 0.0, 0.0, radius_m=15.0, min_lap_time_s=30.0)
    assert len(result.lap_indices) <= 1
