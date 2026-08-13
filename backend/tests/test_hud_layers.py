import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.render.hud_layers import HudRenderer, RenderConfig, _format_lap_time, config_for_resolution


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0:00.00"),
        (5.4, "0:05.40"),
        (61.95, "1:01.95"),  # racing-timing mm:ss.cc, not bare "61.95s"
        (59.998, "1:00.00"),  # centisecond rounding must carry into seconds
        (125.0, "2:05.00"),
    ],
)
def test_format_lap_time(seconds, expected):
    assert _format_lap_time(seconds) == expected


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
    config = RenderConfig(enable_gg=False, enable_minimap=False, width_px=320, height_px=180, dpi=80)
    renderer = HudRenderer(df, lap_indices=[], config=config)
    try:
        assert renderer.ax_spd.get_visible()
        assert not renderer.ax_gg.get_visible()
        assert not renderer.ax_map.get_visible()
        assert not hasattr(renderer, "gg_ball")
    finally:
        renderer.close()


def test_config_for_resolution_matches_real_pixel_size():
    # ffmpeg's overlay filter composites the HUD PNG unscaled -- the
    # canvas must end up pixel-for-pixel the real video's resolution, not
    # just proportionally close to it.
    scaled = config_for_resolution(RenderConfig(), width_px=3840, height_px=2160)
    assert (scaled.width_px, scaled.height_px) == (3840, 2160)


def test_config_for_resolution_keeps_design_width_in_inches():
    # every rect fraction / pt font size in hud_layers.py was tuned against
    # a 16-inch-wide figure (1600px / dpi 100) -- that must stay constant
    # regardless of the real video's resolution, or panels/text would end
    # up a different size relative to the frame at different resolutions.
    base = RenderConfig()
    design_width_in = base.width_px / base.dpi
    for width_px, height_px in [(3840, 2160), (640, 360), (1920, 1080)]:
        scaled = config_for_resolution(base, width_px, height_px)
        assert scaled.width_px / scaled.dpi == pytest.approx(design_width_in, abs=0.05)


def test_config_for_resolution_preserves_other_fields():
    base = RenderConfig(enable_minimap=False, theme="dark_background", max_expected_speed_kmh=200.0)
    scaled = config_for_resolution(base, 800, 450)
    assert scaled.enable_minimap is False
    assert scaled.theme == "dark_background"
    assert scaled.max_expected_speed_kmh == 200.0


def test_draw_frame_out_of_range_is_noop():
    df = _synthetic_df()
    config = RenderConfig(width_px=320, height_px=180, dpi=80)
    renderer = HudRenderer(df, lap_indices=[], config=config)
    try:
        renderer.draw_frame(999)  # should not raise
    finally:
        renderer.close()
