import struct

import numpy as np
import pytest

from app.telemetry.sources.gopro_embedded import (
    convert_speed_to_kmh,
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


@pytest.mark.parametrize(
    "unit,expected_factor",
    [
        ("", 3.6),          # GoPro's own bare-number GPMF convention -> m/s
        ("m/s", 3.6),
        ("km/h", 1.0),
        ("kmh", 1.0),
        ("kph", 1.0),
        ("mph", 1.609344),
        ("kn", 1.852),
        ("knots", 1.852),
        (" KM/H ", 1.0),    # case/whitespace shouldn't matter
    ],
)
def test_convert_speed_to_kmh_uses_the_actual_unit(unit, expected_factor):
    assert convert_speed_to_kmh(10.0, unit) == pytest.approx(10.0 * expected_factor)


def test_convert_speed_to_kmh_unknown_unit_falls_back_to_ms_assumption():
    # an unrecognized unit token still gets converted (not dropped) using
    # the historical m/s assumption, since that's the best guess available
    assert convert_speed_to_kmh(10.0, "furlongs/fortnight") == pytest.approx(36.0)


def _fix_lines(speed_text: str, n: int, lat="37 deg 33' 30.60\" N", lon="5 deg 55' 55.92\" W") -> list[str]:
    lines = []
    for _ in range(n):
        lines += [
            f"GPS Latitude                    : {lat}",
            f"GPS Longitude                   : {lon}",
            f"GPS Speed                       : {speed_text}",
        ]
    return lines


def test_parse_gps_data_regression_unit_suffix_is_not_silently_reinterpreted_as_ms():
    """Regression test for the audit's #1 finding: a GPS Speed line that
    already carries its own unit (as ExifTool prints a standard, non-GoPro
    GPSSpeed tag) must be converted using *that* unit, not blindly
    multiplied by 3.6 as if it were GoPro's unit-less m/s convention -- that
    silent double-conversion is what made speed read ~3.6x too high.
    """
    lines = _fix_lines("42.0 km/h", n=3)
    content = "\n".join(lines)

    df = parse_gps_data(content, duration_sec=10.0)

    assert not df.empty
    assert df["speed"].tolist() == pytest.approx([42.0] * len(df), abs=0.5)


def test_parse_gps_data_bare_number_still_assumed_ms():
    # unchanged behavior for GoPro's own real format: a bare number with no
    # unit suffix is still treated as m/s
    lines = _fix_lines("10.0", n=3)
    content = "\n".join(lines)

    df = parse_gps_data(content, duration_sec=10.0)

    assert not df.empty
    assert df["speed"].tolist() == pytest.approx([36.0] * len(df), abs=0.5)


def test_parse_gps_data_regression_uses_per_block_sample_time():
    """Regression test for the audit's #3 finding: fixes must be placed
    within their own GPMF block's real video-time span (from that block's
    `Sample Time`), not spread evenly across the whole video regardless of
    how fixes are actually distributed across blocks.

    Block 1 (`Sample Time : 0 s`) has 2 fixes; block 2 (`Sample Time : 10 s`)
    has 8 fixes, video duration 20s. The old global-linspace approach would
    place the block boundary at roughly 20 * 1/9 =~ 2.2s -- nowhere near the
    real 10s block boundary the dump actually specifies.
    """
    lines = (
        ["Sample Time                     : 0 s"] + _fix_lines("5.0", n=2)
        + ["Sample Time                     : 10 s"] + _fix_lines("5.0", n=8)
    )
    content = "\n".join(lines)

    df = parse_gps_data(content, duration_sec=20.0)

    assert len(df) == 10
    block_1 = df.iloc[:2]
    block_2 = df.iloc[2:]
    assert block_1["time"].max() < 10.0
    assert block_2["time"].min() == pytest.approx(10.0)
    assert block_2["time"].max() == pytest.approx(20.0)
    assert df["time"].is_monotonic_increasing


def test_parse_gps_data_falls_back_to_global_spacing_without_sample_time():
    # dumps that never carry a `Sample Time` line (e.g. an older exiftool)
    # fall back to exactly the old whole-session-uniform behavior
    lines = _fix_lines("5.0", n=5)
    content = "\n".join(lines)

    df = parse_gps_data(content, duration_sec=10.0)

    assert df["time"].tolist() == pytest.approx(np.linspace(0, 10.0, 5).tolist())


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
