import struct

import numpy as np
import pytest

from app.telemetry.sources.gopro_embedded import (
    dms_to_dd,
    parse_gpmd_accel,
    parse_gps_data,
)


@pytest.mark.parametrize(
    "dms,expected",
    [
        ("37 deg 33' 30.60\" N", pytest.approx(37.5585, abs=1e-4)),
        ("5 deg 55' 55.92\" W", pytest.approx(-5.93220, abs=1e-4)),
        ("0 deg 0' 0.00\" N", pytest.approx(0.0, abs=1e-9)),
    ],
)
def test_dms_to_dd(dms, expected):
    assert dms_to_dd(dms) == expected


def test_dms_to_dd_handles_garbage():
    assert dms_to_dd("") is None
    assert dms_to_dd("not a coordinate") is None


def test_parse_gps_data_against_real_exiftool_dump(gopro_telemetry_txt):
    content = gopro_telemetry_txt.read_text(encoding="utf-8", errors="ignore")
    df = parse_gps_data(content, duration_sec=533.0)

    assert not df.empty
    assert list(df.columns) == ["time", "lat", "lon", "speed"]
    # speeds were converted from m/s to km/h (source file starts near-stationary)
    assert df["speed"].min() >= 0
    # this fixture is a real go-kart session, genuine top speed ~85 km/h --
    # a much tighter ceiling than before is exactly the point: a wide "< 300"
    # ceiling let an unsmoothed single-sample GPS spike through unnoticed
    assert df["speed"].max() < 100
    assert df["time"].is_monotonic_increasing
    assert df["time"].max() == pytest.approx(533.0)
    # real GoPro coordinates from the fixture (~37.558N, ~-5.932W)
    assert df["lat"].between(37.0, 38.0).all()
    assert df["lon"].between(-6.0, -5.0).all()


def test_parse_gps_data_empty_input():
    df = parse_gps_data("no telemetry here", duration_sec=10.0)
    assert df.empty


def _pack_accl_block(samples: list[tuple[int, int, int]]) -> bytes:
    header = b"ACCL" + b"\x00" + bytes([6]) + struct.pack(">H", len(samples))
    body = b"".join(struct.pack(">hhh", *s) for s in samples)
    total = 6 * len(samples)
    pad_len = ((total + 3) & ~3) - total
    return header + body + b"\x00" * pad_len


def test_parse_gpmd_accel_synthetic_block():
    # one_g ~= magnitude of (0, 1000, 0) => 1000
    samples = [(0, 1000, 0), (0, 1000, 100), (0, 1000, -100)] * 20
    blob = _pack_accl_block(samples)

    df = parse_gpmd_accel(blob, duration_sec=1.0, smoothing_window=1)

    assert not df.empty
    assert list(df.columns) == ["time", "lat_g", "lon_g"]
    assert df["lat_g"].mean() == pytest.approx(1.0, abs=0.05)
    assert df["time"].max() == pytest.approx(1.0)


def test_parse_gpmd_accel_no_accl_tag_returns_empty():
    df = parse_gpmd_accel(b"no telemetry markers here at all", duration_sec=1.0)
    assert df.empty
