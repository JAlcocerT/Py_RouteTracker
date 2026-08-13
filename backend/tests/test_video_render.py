import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.core.ffmpeg_utils import get_video_duration
from app.render.hud_layers import RenderConfig
from app.render.video_render import (
    _remap_lap_indices_to_window,
    execute_prepared_render,
    prepare_render_job,
    render_and_composite,
    render_hud_frames,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_remap_lap_indices_to_window_drops_out_of_range_crossings():
    full_df = pd.DataFrame({"time": np.arange(0, 101, 1.0)})
    lap_indices = [10, 50, 90]  # crossings at t=10, t=50, t=90

    window = full_df[(full_df["time"] >= 20) & (full_df["time"] <= 80)].copy()
    window["time"] = window["time"] - 20
    window = window.reset_index(drop=True)

    remapped = _remap_lap_indices_to_window(full_df, lap_indices, trim_start=20, windowed_df=window)

    assert remapped == [30]  # t=50 -> relative time 30 -> position 30 in the windowed df


def test_remap_lap_indices_empty_window_returns_empty():
    full_df = pd.DataFrame({"time": np.arange(0, 10, 1.0)})
    empty_window = pd.DataFrame(columns=["time"])
    assert _remap_lap_indices_to_window(full_df, [1, 2, 3], 0, empty_window) == []


def _synthetic_df(n: int) -> pd.DataFrame:
    t = np.linspace(0, 5, n)
    return pd.DataFrame({
        "time": t,
        "lat": 50.0 + 0.0001 * t,
        "lon": 19.0 + 0.0001 * t,
        "speed": 50 + 5 * np.sin(t),
        "lat_g": 0.0,
        "lon_g": 0.0,
        "lap": 1,
        "last_lap_s": 0.0,
    })


def test_render_hud_frames_writes_one_png_per_row(tmp_path):
    df = _synthetic_df(12)
    config = RenderConfig(width_px=160, height_px=90, dpi=60)

    progress_values = []
    total = render_hud_frames(
        df, lap_indices=[], config=config, frames_dir=tmp_path,
        n_workers=2, on_progress=progress_values.append,
    )

    assert total == 12
    written = sorted(tmp_path.glob("frame_*.png"))
    assert len(written) == 12
    assert progress_values[-1] == pytest.approx(1.0)
    assert progress_values == sorted(progress_values)


def test_render_hud_frames_empty_df_is_noop(tmp_path):
    total = render_hud_frames(pd.DataFrame(columns=["time"]), [], RenderConfig(), tmp_path)
    assert total == 0
    assert list(tmp_path.glob("*.png")) == []


def _make_test_video(path, duration=5, size="160x90", rate=10) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={rate}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_prepare_then_execute_matches_render_and_composite(tmp_path):
    """The split (used by the distributed-worker path) must produce
    equivalent output to the original single-call pipeline (used for local,
    non-distributed rendering) -- same trim window, same frame count,
    same-ish output duration."""
    source = tmp_path / "source.mp4"
    _make_test_video(source, duration=5, rate=10)
    duration = get_video_duration(source)

    df = _synthetic_df(int(duration * 10))
    config = RenderConfig(width_px=160, height_px=90, dpi=60, enable_gg=False, enable_minimap=False)

    combined_out = tmp_path / "combined.mp4"
    combined_progress = []
    render_and_composite(
        source, df, [], config, trim_start=0.0, trim_end=duration,
        work_dir=tmp_path / "combined_work", output_path=combined_out,
        n_workers=1, on_progress=combined_progress.append,
    )

    split_out = tmp_path / "split.mp4"
    split_progress = []
    prepared = prepare_render_job(source, df, [], trim_start=0.0, trim_end=duration, work_dir=tmp_path / "split_work")
    result = execute_prepared_render(prepared, config, split_out, n_workers=1, on_progress=split_progress.append)

    assert combined_out.exists() and combined_out.stat().st_size > 0
    assert split_out.exists() and split_out.stat().st_size > 0
    assert result.frame_count == len(prepared.windowed_telemetry)

    combined_duration = get_video_duration(combined_out)
    split_duration = get_video_duration(split_out)
    assert combined_duration == pytest.approx(split_duration, abs=0.5)

    # both progress sequences must be monotonic and reach 1.0
    for values in (combined_progress, split_progress):
        assert values[-1] == pytest.approx(1.0)
        assert values == sorted(values)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_execute_prepared_render_rescales_canvas_to_real_video_resolution(tmp_path):
    """render_config_from_payload never sets width_px/height_px (see
    app.render.render_config) -- every real render request arrives here
    with RenderConfig's 1600x900 default, regardless of the actual
    footage's resolution. execute_prepared_render must override it to match
    the real (trimmed) video, since ffmpeg's overlay filter composites the
    HUD PNG unscaled, anchored at (0,0)."""
    source = tmp_path / "source.mp4"
    _make_test_video(source, duration=2, size="320x180", rate=10)
    duration = get_video_duration(source)
    df = _synthetic_df(int(duration * 10))
    config = RenderConfig(enable_gg=False, enable_minimap=False)

    prepared = prepare_render_job(source, df, [], trim_start=0.0, trim_end=duration, work_dir=tmp_path / "work")
    out = tmp_path / "out.mp4"
    execute_prepared_render(prepared, config, out, n_workers=1)

    frame_files = sorted((tmp_path / "work" / "hud_frames").glob("frame_*.png"))
    assert frame_files, "expected HUD frames to still be on disk"
    with Image.open(frame_files[0]) as img:
        assert img.size == (320, 180)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_prepare_render_job_output_is_reusable_by_a_different_process_worth_of_state(tmp_path):
    """Simulates the distributed-worker hand-off: prepare on one 'side',
    then execute using only what PreparedRenderJob exposes (trimmed video
    path + a plain telemetry DataFrame + lap_indices + fps) -- nothing else
    from the coordinator's state should be required."""
    source = tmp_path / "source.mp4"
    _make_test_video(source, duration=3, rate=10)
    duration = get_video_duration(source)
    df = _synthetic_df(int(duration * 10))
    config = RenderConfig(width_px=160, height_px=90, dpi=60, enable_gg=False, enable_minimap=False)

    prepared = prepare_render_job(source, df, [], trim_start=0.0, trim_end=duration, work_dir=tmp_path / "work")

    # a "worker" only ever sees these plain values -- round-trip them
    # through dict/DataFrame reconstruction the way JSON transport would
    reconstructed_telemetry = pd.DataFrame(prepared.windowed_telemetry.to_dict(orient="records"))
    from dataclasses import replace
    worker_side_prepared = replace(prepared, windowed_telemetry=reconstructed_telemetry)

    out = tmp_path / "out.mp4"
    result = execute_prepared_render(worker_side_prepared, config, out, n_workers=1)

    assert out.exists() and out.stat().st_size > 0
    assert result.frame_count == len(reconstructed_telemetry)
