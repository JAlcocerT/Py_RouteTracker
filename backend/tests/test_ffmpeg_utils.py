"""Unit coverage for the small ffprobe-output-parsing pieces of
ffmpeg_utils that don't need a real binary/file -- the ffmpeg-dependent
paths (trim, overlay) are exercised for real in test_video_render.py.
"""

import pytest

from app.core import ffmpeg_utils


@pytest.mark.parametrize(
    "raw_output,expected",
    [
        ("30/1", 30.0),
        ("30000/1001", pytest.approx(29.97, abs=0.01)),  # NTSC-style rate
        ("25/1", 25.0),
        ("25", 25.0),  # no denominator at all
    ],
)
def test_get_video_fps_parses_rframerate_fraction(monkeypatch, raw_output, expected):
    monkeypatch.setattr(ffmpeg_utils, "require_binary", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg_utils.subprocess, "check_output",
        lambda cmd: (raw_output + "\n").encode(),
    )

    assert ffmpeg_utils.get_video_fps("unused.mp4") == expected
