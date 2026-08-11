"""Tests for the remote-worker HTTP API (app.api.routes_worker).

Uses a minimal FastAPI app with just the API routers -- deliberately
*without* app.main's lifespan, so LocalRenderWorker isn't running in the
background. These tests drive the claim queue by hand (claim / fetch
inputs / report progress / complete); a competing local worker claiming
jobs before the test can would make this flaky.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes_videos as routes_videos
from app.api import routes_jobs, routes_laps, routes_render, routes_worker
from app.core.state import settings
from app.telemetry.sources.external_gpx import load_gpx_points

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_worker_test_app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(routes_videos.router)
    test_app.include_router(routes_laps.router)
    test_app.include_router(routes_render.router)
    test_app.include_router(routes_jobs.router)
    test_app.include_router(routes_worker.router)
    return test_app


client = TestClient(_make_worker_test_app())

TOKEN = "test-worker-token-abc123"


@pytest.fixture(autouse=True)
def _enable_worker_mode(monkeypatch):
    monkeypatch.setattr(settings, "worker_token", TOKEN)


def _auth(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _test_video_bytes(tmp_path: Path) -> bytes:
    if not HAS_FFMPEG:
        return b"not a real video, extraction never touches this content"
    out = tmp_path / "synthetic.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=25:size=320x180:rate=15",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out.read_bytes()


@pytest.fixture
def uploaded_and_queued_render(monkeypatch, sample_gpx, tmp_path):
    """Uploads a video, waits for extraction, and enqueues a render --
    returns (video_id, job_id) with the render job left `pending` (nothing
    claims it in this test app, since LocalRenderWorker isn't running)."""
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
            data={"source_type": "external_gpx", "video_start_time": video_start_time.isoformat()},
        )
    assert resp.status_code == 200, resp.text
    video_id = resp.json()["video_id"]
    extraction_job_id = resp.json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{extraction_job_id}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    else:
        raise TimeoutError("extraction did not finish in time")

    render_resp = client.post(
        f"/api/videos/{video_id}/render",
        json={"trim_start": 0.0, "trim_end": 20.0, "widgets": {"speedo": True, "gg": False, "minimap": False, "session_graph": False}},
    )
    assert render_resp.status_code == 200, render_resp.text
    return video_id, render_resp.json()["job_id"]


# --- auth / feature-flag behavior ---


def test_worker_endpoints_503_when_token_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "worker_token", "")
    resp = client.get("/api/worker/jobs/next", params={"worker_id": "someone"}, headers=_auth())
    assert resp.status_code == 503


def test_worker_endpoints_401_with_wrong_token():
    resp = client.get("/api/worker/jobs/next", params={"worker_id": "someone"}, headers=_auth("wrong-token"))
    assert resp.status_code == 401


def test_worker_endpoints_401_with_no_auth_header():
    resp = client.get("/api/worker/jobs/next", params={"worker_id": "someone"})
    assert resp.status_code == 401


def test_claim_next_job_204_when_queue_empty():
    resp = client.get("/api/worker/jobs/next", params={"worker_id": "idle-worker"}, headers=_auth())
    assert resp.status_code == 204


# --- full claim -> inputs -> progress -> complete round trip ---


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_full_remote_worker_round_trip(uploaded_and_queued_render, tmp_path):
    video_id, job_id = uploaded_and_queued_render

    claim_resp = client.get("/api/worker/jobs/next", params={"worker_id": "friend-laptop"}, headers=_auth())
    assert claim_resp.status_code == 200, claim_resp.text
    claimed = claim_resp.json()
    assert claimed["job_id"] == job_id
    assert claimed["widgets"]["speedo"] is True
    assert claimed["style"]["theme"] == "cyberpunk"

    # job is now claimed -- a second claim attempt must find nothing
    second_claim = client.get("/api/worker/jobs/next", params={"worker_id": "another-worker"}, headers=_auth())
    assert second_claim.status_code == 204

    job_status = client.get(f"/api/jobs/{job_id}").json()
    assert job_status["status"] == "running"

    video_resp = client.get(f"/api/worker/jobs/{job_id}/inputs/video", headers=_auth())
    assert video_resp.status_code == 200
    assert video_resp.headers["content-type"] == "video/mp4"
    assert len(video_resp.content) > 1000

    telemetry_resp = client.get(f"/api/worker/jobs/{job_id}/inputs/telemetry", headers=_auth())
    assert telemetry_resp.status_code == 200
    manifest = telemetry_resp.json()
    assert len(manifest["points"]) > 0
    assert "fps" in manifest

    progress_resp = client.post(f"/api/worker/jobs/{job_id}/progress", json={"progress": 0.5}, headers=_auth())
    assert progress_resp.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["progress"] == pytest.approx(0.5)

    # simulate the worker having rendered something and uploading it back
    fake_output = tmp_path / "worker_output.mp4"
    fake_output.write_bytes(video_resp.content)  # any valid mp4 bytes will do for this test
    with open(fake_output, "rb") as f:
        complete_resp = client.post(
            f"/api/worker/jobs/{job_id}/complete",
            files={"file": ("output.mp4", f, "video/mp4")},
            headers=_auth(),
        )
    assert complete_resp.status_code == 200, complete_resp.text

    final_status = client.get(f"/api/jobs/{job_id}").json()
    assert final_status["status"] == "done"
    assert final_status["result"]["video_id"] == video_id

    # a remotely-rendered video must be downloadable exactly like a local one
    download_resp = client.get(f"/api/render/{job_id}/download")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "video/mp4"
    assert 'filename="clip_overlay.mp4"' in download_resp.headers["content-disposition"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_worker_can_report_failure(uploaded_and_queued_render):
    video_id, job_id = uploaded_and_queued_render
    client.get("/api/worker/jobs/next", params={"worker_id": "flaky-worker"}, headers=_auth())

    fail_resp = client.post(f"/api/worker/jobs/{job_id}/fail", json={"error": "out of disk space"}, headers=_auth())
    assert fail_resp.status_code == 200

    job_status = client.get(f"/api/jobs/{job_id}").json()
    assert job_status["status"] == "error"
    assert "out of disk space" in job_status["error"]


def test_inputs_404_for_job_that_was_never_claimed():
    resp = client.get("/api/worker/jobs/does-not-exist/inputs/video", headers=_auth())
    assert resp.status_code == 404


def test_progress_404_for_unknown_job():
    resp = client.post("/api/worker/jobs/does-not-exist/progress", json={"progress": 0.5}, headers=_auth())
    assert resp.status_code == 404
