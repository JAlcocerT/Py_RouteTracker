"""End-to-end API wiring tests using the external-GPX source, which needs
no ffmpeg/exiftool binaries for extraction -- only `get_video_duration` (an
ffprobe call) is on this path, and that's monkeypatched out here. This
proves upload -> background extraction -> telemetry -> lap detection -> lap
comparison work end-to-end; the ffmpeg-dependent render step is exercised
separately (skipped locally, run for real in the Docker image -- see
test_video_render.py and webapp/backend/readme.md).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.routes_videos as routes_videos
from app.main import app
from app.telemetry.sources.external_gpx import load_gpx_points

HAS_FFMPEG = shutil.which("ffmpeg") is not None

client = TestClient(app)


def _test_video_bytes(tmp_path: Path) -> bytes:
    """A real (if trivial) mp4 when ffmpeg is on PATH, so the render/trim/
    overlay path gets exercised for real -- not just the extraction path,
    which never reads the video's actual content for external_gpx sources.
    Falls back to placeholder bytes when ffmpeg is absent, where render is
    expected to fail fast on the missing binary regardless of content.
    """
    if not HAS_FFMPEG:
        return b"not a real video, extraction never touches this content"
    out = tmp_path / "synthetic.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=25:size=320x180:rate=15",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out.read_bytes()


def _wait_for_job(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish in time")


@pytest.fixture
def uploaded_video(monkeypatch, sample_gpx, tmp_path):
    monkeypatch.setattr(routes_videos, "get_video_duration", lambda path: 120.0)

    gpx_points = load_gpx_points(sample_gpx)
    video_start_time = gpx_points["timestamp"].iloc[0] - timedelta(seconds=5)

    with open(sample_gpx, "rb") as gpx_file:
        resp = client.post(
            "/api/videos",
            files={
                "video": ("clip.mp4", _test_video_bytes(tmp_path), "video/mp4"),
                "gpx": ("track.gpx", gpx_file.read(), "application/gpx+xml"),
            },
            data={
                "source_type": "external_gpx",
                "video_start_time": video_start_time.isoformat(),
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job = _wait_for_job(body["job_id"])
    assert job["status"] == "done", job
    return body["video_id"]


def test_upload_and_extract_telemetry(uploaded_video):
    resp = client.get(f"/api/videos/{uploaded_video}")
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["telemetry_ready"] is True
    assert meta["source_type"] == "external_gpx"


def test_get_telemetry_points(uploaded_video):
    resp = client.get(f"/api/videos/{uploaded_video}/telemetry")
    assert resp.status_code == 200
    points = resp.json()["points"]
    assert len(points) > 0
    assert set(["time", "lat", "lon", "speed", "lat_g", "lon_g"]).issubset(points[0].keys())


def test_telemetry_not_ready_for_unknown_video():
    resp = client.get("/api/videos/does-not-exist/telemetry")
    assert resp.status_code == 404


def test_detect_and_fetch_laps(uploaded_video):
    resp = client.post(f"/api/videos/{uploaded_video}/laps/detect", json={"start_time_s": 0.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "lap_table" in body

    resp2 = client.get(f"/api/videos/{uploaded_video}/laps")
    assert resp2.status_code == 200
    assert resp2.json()["lap_table"] == body["lap_table"]


def test_laps_compare_before_detection_is_404(uploaded_video):
    resp = client.get(f"/api/videos/{uploaded_video}/laps/compare?lap_a=1&lap_b=2")
    assert resp.status_code == 404


def test_render_job_lifecycle(uploaded_video):
    resp = client.post(
        f"/api/videos/{uploaded_video}/render",
        json={"trim_start": 0.0, "trim_end": 20.0, "widgets": {"speedo": True, "gg": False, "minimap": False, "session_graph": False}},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _wait_for_job(job_id, timeout=60.0)

    if HAS_FFMPEG:
        assert job["status"] == "done", job
        download = client.get(f"/api/render/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == "video/mp4"
    else:
        # no ffmpeg on this machine -- confirm we fail loudly and specifically,
        # not silently or with an unrelated traceback
        assert job["status"] == "error", job
        assert "ffmpeg" in job["error"].lower()


def test_render_rejects_invalid_trim_range(uploaded_video):
    resp = client.post(
        f"/api/videos/{uploaded_video}/render",
        json={"trim_start": 10.0, "trim_end": 5.0},
    )
    assert resp.status_code == 400
