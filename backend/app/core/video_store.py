"""Per-video metadata + cached-artifact bookkeeping.

One JSON file per video under `uploads/<id>/meta.json` -- there's no
multi-user contention to justify a real database here, and JSON is trivial
to inspect by hand while developing.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class VideoMeta:
    id: str
    filename: str
    video_path: str
    source_type: str  # "gopro_embedded" | "external_gpx"
    gpx_path: str | None = None
    duration_sec: float | None = None
    has_accel: bool = False
    telemetry_ready: bool = False
    laps_ready: bool = False
    extraction_job_id: str | None = None
    lap_start_time_s: float | None = None
    # Used by app.core.cleanup's retention sweeper -- uploaded/processed
    # video files are temporary, not permanent storage (see that module).
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    render_job_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class VideoStore:
    def __init__(self, upload_dir: Path, cache_dir: Path):
        self.upload_dir = Path(upload_dir)
        self.cache_dir = Path(cache_dir)
        # Guards read-modify-write of meta.json against concurrent requests
        # for the same video (e.g. two renders queued close together) --
        # without it, one caller's save can silently discard a field change
        # made by the other in between the other's read and write.
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def video_dir(self, video_id: str) -> Path:
        d = self.upload_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def meta_path(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "meta.json"

    def save(self, meta: VideoMeta) -> None:
        self.meta_path(meta.id).write_text(json.dumps(meta.to_json(), indent=2))

    def get(self, video_id: str) -> VideoMeta | None:
        path = self.meta_path(video_id)
        if not path.exists():
            return None
        return VideoMeta(**json.loads(path.read_text()))

    def update(self, video_id: str, **fields) -> VideoMeta:
        with self._lock:
            meta = self.get(video_id)
            if meta is None:
                raise KeyError(video_id)
            for k, v in fields.items():
                setattr(meta, k, v)
            self.save(meta)
            return meta

    def append_render_job_id(self, video_id: str, job_id: str) -> VideoMeta:
        """Atomically appends to render_job_ids under the store's lock.
        Unlike `update()`, whose caller must already hold the field's final
        value, this reads the current list itself right before appending --
        `update(video_id, render_job_ids=[*meta.render_job_ids, job_id])`
        would race two renders queued for the same video close together,
        since both callers could compute their new list from the same
        stale snapshot and the second save would silently drop the first
        job's id."""
        with self._lock:
            meta = self.get(video_id)
            if meta is None:
                raise KeyError(video_id)
            meta.render_job_ids = [*meta.render_job_ids, job_id]
            self.save(meta)
            return meta

    # --- cached artifact paths ---
    def telemetry_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}_telemetry.parquet"

    def laps_annotated_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}_laps_annotated.parquet"

    def laps_table_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}_laps_table.parquet"

    def laps_indices_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}_lap_indices.json"

    def save_lap_indices(self, video_id: str, indices: list[int]) -> None:
        self.laps_indices_path(video_id).write_text(json.dumps(indices))

    def load_lap_indices(self, video_id: str) -> list[int]:
        path = self.laps_indices_path(video_id)
        if not path.exists():
            return []
        return json.loads(path.read_text())
