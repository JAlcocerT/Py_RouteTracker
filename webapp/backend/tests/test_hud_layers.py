import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.render.hud_layers import HudRenderer, RenderConfig


def _synthetic_df(n: int = 20) -> pd.DataFrame:
    t = np.linspace(0, 10, n)
    return pd.DataFrame({
        "time": t,
        "lat": 50.0 + 0.001 * np.sin(t),
        "lon": 19.0 + 0.001 * np.cos(t),
        "speed": 60 + 20 * np.sin(t),
        "lat_g": 0.3 * np.sin(t),
        "lon_g": 0.2 * np.cos(t),
        "lap": (t // 5 + 1).astype(int),
        "last_lap_s": 0.0,
    })


def test_draw_frame_all_widgets_enabled(tmp_path):
    df = _synthetic_df()
    config = RenderConfig(width_px=320, height_px=180, dpi=80)
    renderer = HudRenderer(df, lap_indices=[0, 10], config=config)
    try:
        renderer.draw_frame(5)
        out = tmp_path / "frame.png"
        renderer.save_frame(out)

        assert out.exists() and out.stat().st_size > 0
        img = Image.open(out)
        assert img.mode == "RGBA"
        # background must stay transparent so ffmpeg's overlay filter works
        corner_alpha = img.getpixel((0, 0))[3]
        assert corner_alpha == 0
    finally:
        renderer.close()


def test_disabled_widgets_are_hidden(tmp_path):
    df = _synthetic_df()
    config = RenderConfig(enable_gg=False, enable_minimap=False, enable_session_graph=False, width_px=320, height_px=180, dpi=80)
    renderer = HudRenderer(df, lap_indices=[], config=config)
    try:
        assert renderer.ax_spd.get_visible()
        assert not renderer.ax_gg.get_visible()
        assert not renderer.ax_map.get_visible()
        assert not renderer.ax_gph.get_visible()
        assert not hasattr(renderer, "gg_ball")
    finally:
        renderer.close()


def test_draw_frame_out_of_range_is_noop():
    df = _synthetic_df()
    config = RenderConfig(width_px=320, height_px=180, dpi=80)
    renderer = HudRenderer(df, lap_indices=[], config=config)
    try:
        renderer.draw_frame(999)  # should not raise
    finally:
        renderer.close()
