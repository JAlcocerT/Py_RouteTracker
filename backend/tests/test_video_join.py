"""Coverage for app.core.video_join. The compatibility check and command
construction are pure/mockable and always run; the actual join is only
exercised with a real ffmpeg binary (skipped otherwise, same convention as
test_video_render.py) since it's a real subprocess call.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.core import video_join
from app.core.ffmpeg_utils import get_video_duration, get_video_resolution
from app.core.video_join import (
    IncompatiblePartsError,
    _concat_list_line,
    join_videos,
    probe_stream_signature,
    validate_join_compatible,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_probe_stream_signature_parses_ffprobe_csv(monkeypatch):
    monkeypatch.setattr(video_join, "require_binary", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        video_join.subprocess, "check_output",
        lambda cmd: b"h264,1920,1080,30000/1001\n",
    )

    sig = probe_stream_signature(Path("unused.mp4"))

    assert sig == {"codec": "h264", "width": 1920, "height": 1080, "fps": "30000/1001"}


def test_validate_join_compatible_requires_at_least_two_parts():
    with pytest.raises(IncompatiblePartsError):
        validate_join_compatible([Path("only_one.mp4")])


def test_validate_join_compatible_passes_for_matching_parts(monkeypatch):
    same_sig = {"codec": "h264", "width": 1920, "height": 1080, "fps": "30/1"}
    monkeypatch.setattr(video_join, "probe_stream_signature", lambda p: same_sig)

    validate_join_compatible([Path("GH010001.MP4"), Path("GH020001.MP4"), Path("GH030001.MP4")])


def test_validate_join_compatible_raises_for_mismatched_resolution(monkeypatch):
    sigs = {
        "GH010001.MP4": {"codec": "h264", "width": 1920, "height": 1080, "fps": "30/1"},
        "GH020001.MP4": {"codec": "h264", "width": 1280, "height": 720, "fps": "30/1"},
    }
    monkeypatch.setattr(video_join, "probe_stream_signature", lambda p: sigs[p.name])

    with pytest.raises(IncompatiblePartsError, match="GH020001.MP4"):
        validate_join_compatible([Path("GH010001.MP4"), Path("GH020001.MP4")])


def test_concat_list_line_escapes_single_quotes():
    line = _concat_list_line(Path("/videos/it's a clip.mp4"))

    assert line == "file '/videos/it'\\''s a clip.mp4'\n"


def test_join_videos_invokes_ffmpeg_with_full_stream_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(video_join, "require_binary", lambda name: "/usr/bin/ffmpeg")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # The concat list file must exist (with both parts listed) at the
        # moment ffmpeg would actually read it.
        list_path = Path(cmd[cmd.index("-i") + 1])
        captured["list_contents"] = list_path.read_text()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(video_join.subprocess, "run", fake_run)

    part_a = tmp_path / "GH010001.MP4"
    part_b = tmp_path / "GH020001.MP4"
    part_a.write_bytes(b"")
    part_b.write_bytes(b"")
    dest = tmp_path / "out" / "joined.mp4"

    join_videos([part_a, part_b], dest)

    cmd = captured["cmd"]
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "0"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert "-copy_unknown" in cmd
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat"
    assert cmd[-1] == str(dest)
    assert "GH010001.MP4" in captured["list_contents"]
    assert "GH020001.MP4" in captured["list_contents"]
    # the temp concat-list file is cleaned up after the call
    assert list(dest.parent.glob("*_parts.txt")) == []


def _make_test_video(path: Path, duration: float, size: str = "160x90", rate: int = 10) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={rate}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_join_videos_produces_lossless_concatenation(tmp_path):
    """Two codec/resolution-identical synthetic parts (standing in for
    chapters of one real recording) should join into a single file whose
    duration is the sum of both parts, with the resolution unchanged --
    proof the concat demuxer's stream copy actually ran end to end."""
    part_a = tmp_path / "GH010001.MP4"
    part_b = tmp_path / "GH020001.MP4"
    _make_test_video(part_a, duration=3)
    _make_test_video(part_b, duration=2)

    validate_join_compatible([part_a, part_b])  # should not raise

    dest = tmp_path / "joined.mp4"
    join_videos([part_a, part_b], dest)

    assert dest.exists() and dest.stat().st_size > 0
    assert get_video_duration(dest) == pytest.approx(5.0, abs=0.3)
    assert get_video_resolution(dest) == (160, 90)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_validate_join_compatible_rejects_real_mismatched_parts(tmp_path):
    part_a = tmp_path / "a.mp4"
    part_b = tmp_path / "b.mp4"
    _make_test_video(part_a, duration=2, size="160x90")
    _make_test_video(part_b, duration=2, size="320x180")

    with pytest.raises(IncompatiblePartsError):
        validate_join_compatible([part_a, part_b])
