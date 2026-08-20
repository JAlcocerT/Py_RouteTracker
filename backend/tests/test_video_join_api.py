"""Wiring tests for POST /api/videos/join and GET /api/videos/{id}/source.

Request-shape validation (part count, source_type/gpx combination) happens
before any ffprobe/ffmpeg call, so those cases run unconditionally; the
real join itself needs a real ffmpeg and is skipped otherwise, same
convention as test_api.py / test_video_render.py.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

HAS_FFMPEG = shutil.which("ffmpeg") is not None

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _run_app_lifespan():
    with client:
        yield


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


def _synthetic_part_bytes(tmp_path: Path, name: str, duration: float) -> bytes:
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=160x90:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out.read_bytes()


def test_join_rejects_fewer_than_two_parts():
    resp = client.post(
        "/api/videos/join",
        files=[("videos", ("GH010001.MP4", b"only one part", "video/mp4"))],
        data={"source_type": "gopro_embedded"},
    )
    assert resp.status_code == 400
    assert "at least two" in resp.json()["detail"]


def test_join_rejects_invalid_source_type():
    resp = client.post(
        "/api/videos/join",
        files=[
            ("videos", ("GH010001.MP4", b"a", "video/mp4")),
            ("videos", ("GH020001.MP4", b"b", "video/mp4")),
        ],
        data={"source_type": "not_a_real_source"},
    )
    assert resp.status_code == 400


def test_join_requires_gpx_for_external_source():
    resp = client.post(
        "/api/videos/join",
        files=[
            ("videos", ("part1.mp4", b"a", "video/mp4")),
            ("videos", ("part2.mp4", b"b", "video/mp4")),
        ],
        data={"source_type": "external_gpx"},
    )
    assert resp.status_code == 400
    assert "gpx" in resp.json()["detail"]


def test_source_endpoint_404s_for_unknown_video():
    resp = client.get("/api/videos/does-not-exist/source")
    assert resp.status_code == 404


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_join_two_real_parts_then_fetch_joined_source(tmp_path):
    part_a = _synthetic_part_bytes(tmp_path, "GH010001.MP4", duration=3)
    part_b = _synthetic_part_bytes(tmp_path, "GH020001.MP4", duration=2)

    resp = client.post(
        "/api/videos/join",
        files=[
            ("videos", ("GH010001.MP4", part_a, "video/mp4")),
            ("videos", ("GH020001.MP4", part_b, "video/mp4")),
        ],
        data={"source_type": "gopro_embedded"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["duration_sec"] == pytest.approx(5.0, abs=0.3)

    # extraction itself is expected to fail (no real GPS/GPMD in a
    # synthetic testsrc clip) -- what this test is actually proving is
    # that the join step succeeded and the joined file is servable, not
    # that fake footage carries real telemetry.
    job = _wait_for_job(body["job_id"])
    assert job["status"] == "error"

    source_resp = client.get(f"/api/videos/{body['video_id']}/source")
    assert source_resp.status_code == 200
    assert source_resp.headers["content-type"] == "video/mp4"
    assert len(source_resp.content) > 0
