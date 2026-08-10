from datetime import datetime, timedelta, timezone

from app.core.cleanup import delete_video_artifacts, sweep_expired_videos
from app.core.config import Settings
from app.core.jobs import JobManager
from app.core.video_store import VideoMeta, VideoStore


def _make_settings(tmp_path, retention_minutes=60) -> Settings:
    settings = Settings.__new__(Settings)  # bypass __init__'s env-var reads
    settings.data_dir = tmp_path
    settings.upload_dir = tmp_path / "uploads"
    settings.cache_dir = tmp_path / "cache"
    settings.output_dir = tmp_path / "outputs"
    settings.work_dir = tmp_path / "work"
    settings.jobs_db_path = tmp_path / "jobs.db"
    settings.retention_minutes = retention_minutes
    for d in (settings.upload_dir, settings.cache_dir, settings.output_dir, settings.work_dir):
        d.mkdir(parents=True, exist_ok=True)
    return settings


def _seed_video(store: VideoStore, settings: Settings, video_id: str, age_minutes: float, extraction_job_id: str | None = None, render_job_ids=None) -> None:
    video_dir = store.video_dir(video_id)
    (video_dir / "clip.mp4").write_bytes(b"fake video bytes")
    (video_dir / "track.gpx").write_text("<gpx></gpx>")

    created_at = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    meta = VideoMeta(
        id=video_id,
        filename="clip.mp4",
        video_path=str(video_dir / "clip.mp4"),
        source_type="external_gpx",
        gpx_path=str(video_dir / "track.gpx"),
        created_at=created_at,
        extraction_job_id=extraction_job_id,
        render_job_ids=render_job_ids or [],
    )
    store.save(meta)

    store.telemetry_path(video_id).write_text("fake parquet")
    store.laps_annotated_path(video_id).write_text("fake parquet")
    settings.output_dir.joinpath(f"{video_id}_somejob.mp4").write_bytes(b"fake output")


def test_delete_video_artifacts_removes_everything(tmp_path):
    settings = _make_settings(tmp_path)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    _seed_video(store, settings, "vid-1", age_minutes=0)

    delete_video_artifacts(store, settings, "vid-1")

    assert not (settings.upload_dir / "vid-1").exists()
    assert not store.telemetry_path("vid-1").exists()
    assert not store.laps_annotated_path("vid-1").exists()
    assert list(settings.output_dir.glob("vid-1_*.mp4")) == []


def test_delete_video_artifacts_is_idempotent(tmp_path):
    settings = _make_settings(tmp_path)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    # never seeded -- nothing exists on disk for this id
    delete_video_artifacts(store, settings, "does-not-exist")  # must not raise


def test_sweep_deletes_only_expired_videos(tmp_path):
    settings = _make_settings(tmp_path, retention_minutes=60)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    job_manager = JobManager(tmp_path / "jobs.db")

    _seed_video(store, settings, "old-video", age_minutes=120)
    _seed_video(store, settings, "fresh-video", age_minutes=5)

    cleaned = sweep_expired_videos(store, settings, job_manager)

    assert cleaned == 1
    assert not (settings.upload_dir / "old-video").exists()
    assert (settings.upload_dir / "fresh-video").exists()


def test_sweep_skips_video_with_active_extraction_job(tmp_path):
    settings = _make_settings(tmp_path, retention_minutes=60)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    job_manager = JobManager(tmp_path / "jobs.db")

    job_id = job_manager.create_job("extract_telemetry")  # left in "pending" -- never submitted
    _seed_video(store, settings, "expired-but-active", age_minutes=120, extraction_job_id=job_id)

    cleaned = sweep_expired_videos(store, settings, job_manager)

    assert cleaned == 0
    assert (settings.upload_dir / "expired-but-active").exists()


def test_sweep_skips_video_with_active_render_job(tmp_path):
    settings = _make_settings(tmp_path, retention_minutes=60)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    job_manager = JobManager(tmp_path / "jobs.db")

    render_job_id = job_manager.create_job("render")
    _seed_video(store, settings, "rendering-now", age_minutes=120, render_job_ids=[render_job_id])

    cleaned = sweep_expired_videos(store, settings, job_manager)

    assert cleaned == 0
    assert (settings.upload_dir / "rendering-now").exists()


def test_sweep_deletes_once_render_job_finishes(tmp_path):
    settings = _make_settings(tmp_path, retention_minutes=60)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    job_manager = JobManager(tmp_path / "jobs.db")

    render_job_id = job_manager.create_job("render")
    job_manager.submit(render_job_id, lambda progress_cb: {"ok": True})
    import time
    for _ in range(100):
        if job_manager.get_job(render_job_id).status == "done":
            break
        time.sleep(0.02)

    _seed_video(store, settings, "finished-render", age_minutes=120, render_job_ids=[render_job_id])

    cleaned = sweep_expired_videos(store, settings, job_manager)

    assert cleaned == 1
    assert not (settings.upload_dir / "finished-render").exists()


def test_sweep_empty_upload_dir_is_noop(tmp_path):
    settings = _make_settings(tmp_path)
    store = VideoStore(settings.upload_dir, settings.cache_dir)
    job_manager = JobManager(tmp_path / "jobs.db")
    assert sweep_expired_videos(store, settings, job_manager) == 0
