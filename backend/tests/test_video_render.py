import shutil

import numpy as np
import pandas as pd
import pytest

from app.render.hud_layers import RenderConfig
from app.render.video_render import _remap_lap_indices_to_window, render_hud_frames

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


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment; exercised in the Docker image instead")
def test_trim_and_overlay_require_ffmpeg():
    pytest.skip("placeholder for the Docker-environment integration pass (see webapp/backend/readme.md)")
