"""Remote render worker: run this on any machine -- not just the
coordinator's homelab box -- to contribute rendering compute. Uses the same
rendering code as the coordinator's built-in local worker
(app.render.local_worker); the only difference is that it polls a
coordinator over HTTP instead of claiming jobs from the local database
directly. Deliberately standalone: never touches app.core.config's
`settings` singleton or the coordinator's data directory -- everything it
needs comes from the coordinator over the wire, into its own temp dir.

Usage:
    python -m app.worker_main --server http://<coordinator-host>:7000 --token <token>

See backend/readme.md's "Distributed rendering" section for setup and the
security model (short version: the token is a shared secret -- anyone
holding it can claim, and briefly receive, any pending render on that
coordinator; only run this pointed at a coordinator you trust, with a token
you control).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.core.binaries import require_binary
from app.render.render_config import render_config_from_payload
from app.render.video_render import PreparedRenderJob, execute_prepared_render

_DEFAULT_MAX_RENDER_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))


def _log(name: str, message: str) -> None:
    print(f"[worker:{name}] {message}", flush=True)


def _claim_job(client: httpx.Client, worker_name: str) -> dict[str, Any] | None:
    resp = client.get("/api/worker/jobs/next", params={"worker_id": worker_name})
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def _download_to(client: httpx.Client, path: str, dest: Path) -> None:
    with client.stream("GET", path) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)


def _process_job(client: httpx.Client, job: dict[str, Any], max_render_workers: int) -> None:
    job_id = job["job_id"]

    with tempfile.TemporaryDirectory(prefix=f"routetracker_worker_{job_id}_") as tmp:
        work_dir = Path(tmp)
        video_path = work_dir / "trimmed.mp4"
        _download_to(client, f"/api/worker/jobs/{job_id}/inputs/video", video_path)

        telemetry_resp = client.get(f"/api/worker/jobs/{job_id}/inputs/telemetry")
        telemetry_resp.raise_for_status()
        manifest = telemetry_resp.json()

        prepared = PreparedRenderJob(
            trimmed_video_path=video_path,
            windowed_telemetry=pd.DataFrame(manifest["points"]),
            lap_indices=manifest["lap_indices"],
            fps=manifest["fps"],
            work_dir=work_dir,
        )
        config = render_config_from_payload(job)
        output_path = work_dir / "output.mp4"

        last_reported = -1.0

        def on_progress(p: float) -> None:
            nonlocal last_reported
            # Throttle: don't hammer the coordinator with a request per
            # frame-chunk update. A missed update is harmless -- the next
            # one supersedes it, and it's also the claim's lease heartbeat
            # (see StaleRenderJobRequeuer), so we do always send the final 1.0.
            if p < 1.0 and p - last_reported < 0.02:
                return
            last_reported = p
            try:
                client.post(f"/api/worker/jobs/{job_id}/progress", json={"progress": p})
            except httpx.HTTPError:
                pass

        execute_prepared_render(prepared, config, output_path, n_workers=max_render_workers, on_progress=on_progress)

        with open(output_path, "rb") as f:
            resp = client.post(f"/api/worker/jobs/{job_id}/complete", files={"file": ("output.mp4", f, "video/mp4")})
            resp.raise_for_status()


def run_worker(server: str, token: str, name: str, poll_interval: float, max_render_workers: int) -> None:
    client = httpx.Client(
        base_url=server.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(30.0, read=120.0),
    )
    _log(name, f"polling {server} every {poll_interval}s (max_render_workers={max_render_workers})")

    while True:
        try:
            job = _claim_job(client, name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 503):
                _log(name, f"coordinator rejected us: {exc.response.status_code} {exc.response.text} -- check --token / ROUTETRACKER_WORKER_TOKEN")
                raise SystemExit(1) from exc
            _log(name, f"coordinator error ({exc}); retrying in {poll_interval}s")
            time.sleep(poll_interval)
            continue
        except httpx.HTTPError as exc:
            _log(name, f"coordinator unreachable ({exc}); retrying in {poll_interval}s")
            time.sleep(poll_interval)
            continue

        if job is None:
            time.sleep(poll_interval)
            continue

        _log(name, f"claimed job {job['job_id']}")
        try:
            _process_job(client, job, max_render_workers)
            _log(name, f"job {job['job_id']} complete")
        except Exception as exc:  # noqa: BLE001 - must not crash the poll loop over one bad job
            _log(name, f"job {job['job_id']} failed: {exc}")
            try:
                client.post(f"/api/worker/jobs/{job['job_id']}/fail", json={"error": str(exc)})
            except httpx.HTTPError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="Coordinator base URL, e.g. http://100.x.x.x:7000")
    parser.add_argument("--token", default=os.environ.get("ROUTETRACKER_WORKER_TOKEN", ""), help="Must match the coordinator's ROUTETRACKER_WORKER_TOKEN (or set that env var instead)")
    parser.add_argument("--name", default=socket.gethostname(), help="Identifies this worker in logs/job history (default: hostname)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls when idle (default: 5)")
    parser.add_argument("--max-render-workers", type=int, default=_DEFAULT_MAX_RENDER_WORKERS, help=f"Frame-render parallelism on this machine (default: {_DEFAULT_MAX_RENDER_WORKERS})")
    args = parser.parse_args(argv)

    if not args.token:
        print("error: --token (or ROUTETRACKER_WORKER_TOKEN) is required", file=sys.stderr)
        return 1

    try:
        require_binary("ffmpeg")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        run_worker(args.server, args.token, args.name, args.poll_interval, args.max_render_workers)
    except KeyboardInterrupt:
        print("\nworker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
