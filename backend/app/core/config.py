from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        data_dir = Path(os.environ.get("ROUTETRACKER_DATA_DIR", "./data")).resolve()
        self.data_dir = data_dir
        self.upload_dir = data_dir / "uploads"
        self.cache_dir = data_dir / "cache"
        self.output_dir = data_dir / "outputs"
        self.work_dir = data_dir / "work"
        self.jobs_db_path = data_dir / "jobs.db"
        self.default_target_fps = float(os.environ.get("ROUTETRACKER_TARGET_FPS", "30"))
        # Each render worker is a full Python + matplotlib process (~150-250MB
        # baseline). `cpu_count() - 1` is fine on a dedicated box, but this is
        # meant to run on shared/non-dedicated hardware (a homelab box also
        # doing other things) -- defaulting to every-core-but-one risks
        # starving whatever else is running and can itself add memory
        # pressure on constrained (e.g. 16GB) hosts. Capped at 4 by default;
        # override via env var if you know you have the headroom.
        default_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        self.max_render_workers = int(os.environ.get("ROUTETRACKER_MAX_RENDER_WORKERS", str(default_workers)))

        # This is a local tool, not video storage -- uploaded source videos
        # and rendered output are only ever meant to live on disk long
        # enough to download the result. app.core.cleanup deletes each
        # video's files once they're older than this many minutes.
        self.retention_minutes = float(os.environ.get("ROUTETRACKER_RETENTION_MINUTES", "60"))
        self.sweep_interval_seconds = float(os.environ.get("ROUTETRACKER_SWEEP_INTERVAL_SECONDS", "300"))

        # Distributed rendering (app.api.routes_worker, app.worker_main) is
        # off by default -- an empty token disables the whole /api/worker
        # surface (503s), since a shared-secret worker token that's easy to
        # forget to set would otherwise mean "anyone who can reach this
        # server can claim any pending render." Generate one with e.g.
        # `openssl rand -hex 32`.
        self.worker_token = os.environ.get("ROUTETRACKER_WORKER_TOKEN", "").strip()
        self.worker_lease_minutes = float(os.environ.get("ROUTETRACKER_WORKER_LEASE_MINUTES", "30"))

        # How long a freshly-queued render job is held back from "claim
        # whatever's next" (the built-in local worker, and any standing
        # remote workers) so the uploader has a real chance to self-render
        # it first via the per-job claim_token shown in the UI -- see
        # app.core.jobs.JobManager.claim_next's min_age_seconds. Without
        # this, the local worker's own ~2s poll loop wins that race almost
        # every time, since no human can copy-paste a command that fast.
        self.self_render_grace_seconds = float(os.environ.get("ROUTETRACKER_SELF_RENDER_GRACE_SECONDS", "15"))

        for d in (self.upload_dir, self.cache_dir, self.output_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
