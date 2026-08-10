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
        self.max_render_workers = int(os.environ.get("ROUTETRACKER_MAX_RENDER_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))

        for d in (self.upload_dir, self.cache_dir, self.output_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
