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

        for d in (self.upload_dir, self.cache_dir, self.output_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
