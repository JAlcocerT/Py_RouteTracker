from datetime import timedelta

import pytest

from app.telemetry.sources.external_gpx import ExternalGpxSource, compute_speed_kmh, load_gpx_points


def test_load_gpx_points_against_real_fixture(sample_gpx):
    df = load_gpx_points(sample_gpx)

    assert not df.empty
    assert set(["timestamp", "lat", "lon", "ele"]).issubset(df.columns)
    assert df["timestamp"].is_monotonic_increasing
    # sample route is in southern Poland
    assert df["lat"].between(49.0, 51.0).all()
    assert df["lon"].between(18.0, 21.0).all()


def test_compute_speed_kmh_is_nonnegative(sample_gpx):
    df = load_gpx_points(sample_gpx)
    speed = compute_speed_kmh(df)
    assert (speed >= 0).all()
    # no prior point to derive speed from at index 0 -- the median filter
    # blends that 0 sentinel with its neighbor, so it's small, not exactly 0
    assert speed.iloc[0] < speed.max()


def test_external_gpx_source_aligns_to_video_start_time(sample_gpx):
    df = load_gpx_points(sample_gpx)
    first_point_time = df["timestamp"].iloc[0]
    # pretend the video started 10s before the first GPX point
    video_start = first_point_time - timedelta(seconds=10)

    source = ExternalGpxSource(sample_gpx, target_fps=1.0, video_start_time=video_start)
    result = source.extract(video_path="unused.mp4", duration_sec=120.0)

    assert result.source_name == "external_gpx"
    assert not result.has_accel
    assert not result.df.empty
    assert result.df["time"].min() >= 0
    assert result.df["time"].max() <= 120.0


def test_external_gpx_source_empty_when_window_excludes_all_points(sample_gpx):
    df = load_gpx_points(sample_gpx)
    # video starts a full day after the GPX track -> no overlap
    video_start = df["timestamp"].iloc[0] + timedelta(days=1)

    source = ExternalGpxSource(sample_gpx, target_fps=1.0, video_start_time=video_start)
    result = source.extract(video_path="unused.mp4", duration_sec=60.0)

    assert result.df.empty
