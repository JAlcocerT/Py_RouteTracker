"""HUD drawing: ported from legacy/overlay/racing_hud_v7.py:252-338.

Three real changes from the original:
  1. Every widget (speed arc, G-G diagram, minimap, session graph) is
     independently toggle-able via RenderConfig instead of always-on.
  2. The figure is drawn with a transparent background (`transparent=True`
     on savefig, no `fig.patch.set_facecolor('black')`) and frames are
     saved as individual PNGs rather than encoded straight to video —
     video_render.py composites this PNG sequence onto the real footage
     via ffmpeg's `overlay` filter, closing the gap where v7 only ever
     produced a standalone HUD clip.
  3. Frames are captured via real matplotlib blitting instead of
     `Figure.savefig`. v7's own `update()` already returned exactly the
     tuple of changed artists FuncAnimation's blitting API expects — but
     that was cosmetic: `Animation.save()` always calls `Figure.savefig`
     per frame regardless of `blit=True`, which redoes a full figure
     layout/redraw (axes, ticks, the static route line, glow effects, ...)
     on every single frame. That's the actual reason renders were so slow
     (overlay/comparison.md logs 14+ minutes for one clip). Here, the
     static background (everything that doesn't change frame-to-frame) is
     rendered and cached once via `copy_from_bbox`; each frame only
     restores that cached region and redraws the handful of artists that
     actually moved (`draw_artist` + `blit`), then the canvas' raw RGBA
     buffer is saved directly with PIL — skipping `savefig`'s relayout
     entirely. Same pixels, same DPI, same visual output; just far less
     redundant work per frame. Combined with the multiprocessing
     frame-chunking in video_render.py, this is the main lever for making
     renders usable on modest, shared/non-dedicated hardware.

This module renders ONE frame at a time (`draw_frame`) rather than using
matplotlib.animation.FuncAnimation — FuncAnimation buys nothing once
video_render.py is doing the parallelization and file-writing itself, and
manual blitting needs direct control over the restore/draw/blit sequence
anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

try:
    import mplcyberpunk
    _HAS_CYBERPUNK = True
except Exception:
    # mplcyberpunk pokes at matplotlib's private style-loading internals at
    # import time, which has broken across matplotlib releases before ours;
    # fall back to the built-in dark_background theme rather than crash.
    _HAS_CYBERPUNK = False


@dataclass
class RenderConfig:
    enable_speedo: bool = True
    enable_gg: bool = True
    enable_minimap: bool = True
    enable_session_graph: bool = True
    max_expected_speed_kmh: float = 85.0
    limit_g: float = 1.5
    theme: str = "cyberpunk"  # "cyberpunk" or "dark_background"
    width_px: int = 1600
    height_px: int = 900
    dpi: int = 100
    gg_trail_frames: int = 15
    minimap_trail_frames: int = 150


# Every rect fraction and pt-sized font in this module was tuned by eye
# against the default 1600x900 @ dpi 100 canvas -- i.e. a 16x9-inch figure.
# That's the "design width" config_for_resolution scales dpi against.
_DESIGN_WIDTH_IN = RenderConfig().width_px / RenderConfig().dpi


def config_for_resolution(base: RenderConfig, width_px: int, height_px: int) -> RenderConfig:
    """Returns a copy of `base` re-targeted at a video's real pixel
    resolution, keeping every panel/font visually proportional regardless of
    that resolution.

    This matters because ffmpeg's overlay filter (app.core.ffmpeg_utils.
    overlay_png_sequence) composites the HUD PNG sequence at its own native
    pixel size, unscaled, anchored at (0,0) -- it does not stretch the HUD
    to fit the footage. Without this, corner-anchored panels only actually
    land in the real video frame's corners by coincidence (i.e. only when
    the footage happens to be exactly 1600x900).

    dpi is derived from width alone, not width and height independently, so
    the figure's physical design width stays fixed at _DESIGN_WIDTH_IN
    inches; every rect fraction and font size in this module was tuned
    against that. Real action-cam footage is essentially always landscape,
    so anchoring on width is the reasonable general case -- an unusually
    tall/narrow (portrait) input still renders (matplotlib just gets a
    non-9-inch-tall figure), just with text sized relative to width rather
    than height.
    """
    dpi = max(1, round(width_px / _DESIGN_WIDTH_IN))
    return replace(base, width_px=width_px, height_px=height_px, dpi=dpi)


_OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]

# Rounded, semi-transparent dark backing panels give every widget contrast
# against arbitrary, unpredictable footage (bright sky, glare, light-colored
# kerbs/barriers) that a bare glow/outline alone can't reliably beat -- see
# the mockups compared against a synthetic busy background before this
# layout was picked. Corner placement (bottom-left/right, tied together by
# a thin strip along the very bottom edge) keeps the center of the frame,
# where the actual footage subject usually is, uncovered.
_PANEL_FACE_RGB = (0.02, 0.05, 0.06)
_PANEL_EDGE = (1, 1, 1, 0.18)

# Figure-fraction (left, bottom, width, height) / [left, bottom, width,
# height] rects. Kept as module constants rather than RenderConfig fields --
# this is a considered visual design, not a per-render knob callers should
# reasonably want to override (unlike the toggles/thresholds on
# RenderConfig itself).
_LEFT_PANEL_RECT = (0.015, 0.03, 0.30, 0.40)
_RIGHT_PANEL_RECT = (0.685, 0.03, 0.30, 0.40)
_STRIP_RECT = (0.015, 0.445, 0.97, 0.09)
_SPEEDO_RECT = [0.02, 0.20, 0.29, 0.27]
_GG_RECT = [0.045, 0.045, 0.10, 0.14]
_MINIMAP_RECT = [0.71, 0.06, 0.25, 0.36]
_SESSION_RECT = [0.03, 0.455, 0.94, 0.07]


class HudRenderer:
    """Draws one frame of the telemetry HUD onto a matplotlib figure.

    `df` must already be windowed to the render range and annotated with
    `lap` / `last_lap_s` (see app.laps.detection.detect_laps) — this class
    only draws, it doesn't know about trimming or lap math.
    """

    def __init__(self, df: pd.DataFrame, lap_indices: list[int], config: RenderConfig):
        self.df = df.reset_index(drop=True)
        self.lap_indices = [i for i in lap_indices if 0 <= i < len(self.df)]
        self.config = config
        self._dynamic_artists: list = []
        self._build_figure()
        # First full draw renders (and lets us cache) everything static --
        # axes, static route/session lines, glow effects -- while the
        # dynamic artists above are still in their empty initial state.
        self.fig.canvas.draw()
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _build_figure(self) -> None:
        cfg = self.config
        style = "cyberpunk" if (cfg.theme == "cyberpunk" and _HAS_CYBERPUNK) else "dark_background"
        plt.style.use(style)

        fig_w, fig_h = cfg.width_px / cfg.dpi, cfg.height_px / cfg.dpi
        self.fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.dpi)
        self.fig.patch.set_alpha(0.0)

        self._build_panels()

        self.ax_spd = self.fig.add_axes(_SPEEDO_RECT)
        self.ax_gg = self.fig.add_axes(_GG_RECT)
        self.ax_map = self.fig.add_axes(_MINIMAP_RECT)
        self.ax_gph = self.fig.add_axes(_SESSION_RECT)

        for ax, enabled in (
            (self.ax_spd, cfg.enable_speedo),
            (self.ax_gg, cfg.enable_gg),
            (self.ax_map, cfg.enable_minimap),
            (self.ax_gph, cfg.enable_session_graph),
        ):
            ax.patch.set_alpha(0.0)
            ax.axis("off")
            ax.set_visible(enabled)

        if cfg.enable_speedo:
            self._build_speedo()
        if cfg.enable_gg:
            self._build_gg()
        if cfg.enable_minimap:
            self._build_minimap()
        if cfg.enable_session_graph:
            self._build_session_graph()

        if _HAS_CYBERPUNK and style == "cyberpunk" and cfg.enable_speedo:
            # add_glow_effects() also calls add_underglow(), which fills the
            # area under every line down to y=0 -- fine for a time series,
            # but this axes' only line at build time is the static full-arc
            # gauge track, so that fill becomes a translucent wedge sitting
            # right behind the speed readout. Only the line-glow half is
            # wanted here.
            mplcyberpunk.make_lines_glow(ax=self.ax_spd)

    def _build_panels(self) -> None:
        """Static (never redrawn per-frame) backing panels -- part of the
        cached background like the rest of a widget's non-moving pixels.
        Added to the figure before any widget axes so they draw underneath."""
        cfg = self.config
        left_needed = cfg.enable_speedo or cfg.enable_gg
        right_needed = cfg.enable_minimap
        strip_needed = cfg.enable_session_graph
        if not (left_needed or right_needed or strip_needed):
            return

        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.patch.set_alpha(0)
        if left_needed:
            ax.add_patch(self._panel_patch(_LEFT_PANEL_RECT))
        if right_needed:
            ax.add_patch(self._panel_patch(_RIGHT_PANEL_RECT))
        if strip_needed:
            ax.add_patch(self._panel_patch(_STRIP_RECT, radius=0.015, alpha=0.35))

    @staticmethod
    def _panel_patch(rect: tuple[float, float, float, float], radius: float = 0.02, alpha: float = 0.42) -> FancyBboxPatch:
        left, bottom, width, height = rect
        return FancyBboxPatch(
            (left, bottom), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=1.2, edgecolor=_PANEL_EDGE, facecolor=(*_PANEL_FACE_RGB, alpha),
        )

    def _build_speedo(self) -> None:
        theta = np.linspace(np.pi, 0, 100)
        rad = 0.36
        self._arc_x = 0.5 + rad * np.cos(theta)
        self._arc_y = 0.32 + rad * np.sin(theta)
        self.ax_spd.plot(self._arc_x, self._arc_y, color="white", lw=2, alpha=0.15)
        self.sp_arc, = self.ax_spd.plot([], [], lw=9, solid_capstyle="round", path_effects=_OUTLINE)
        self.sp_txt = self.ax_spd.text(0.5, 0.30, "", fontsize=42, color="white", ha="center", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.text(0.5, 0.16, "KM/H", fontsize=13, color="#00ff9f", ha="center", fontweight="bold", path_effects=_OUTLINE)
        self.lap_txt = self.ax_spd.text(0.06, 0.90, "LAP", fontsize=16, color="cyan", ha="left", fontweight="bold", path_effects=_OUTLINE)
        self.last_txt = self.ax_spd.text(0.94, 0.90, "LAST", fontsize=13, color="yellow", ha="right", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.set_xlim(0, 1)
        self.ax_spd.set_ylim(0, 1)
        self._dynamic_artists += [self.sp_arc, self.sp_txt, self.lap_txt, self.last_txt]

    def _build_gg(self) -> None:
        cfg = self.config
        self.ax_gg.set_xlim(-cfg.limit_g, cfg.limit_g)
        self.ax_gg.set_ylim(-cfg.limit_g, cfg.limit_g)
        self.ax_gg.set_aspect("equal")
        self.ax_gg.add_artist(plt.Circle((0, 0), 0.5, color="white", fill=False, alpha=0.25, ls="--"))
        self.ax_gg.add_artist(plt.Circle((0, 0), 1.0, color="white", fill=False, alpha=0.4, ls="-"))
        self.ax_gg.axhline(0, color="white", alpha=0.1)
        self.ax_gg.axvline(0, color="white", alpha=0.1)
        self.gg_trail, = self.ax_gg.plot([], [], color="cyan", lw=2, alpha=0.6, path_effects=_OUTLINE)
        self.gg_ball, = self.ax_gg.plot([], [], "o", color="#ff0055", markersize=11, mec="white", zorder=10)
        self.gg_txt = self.ax_gg.text(0.05, 0.85, "", transform=self.ax_gg.transAxes, color="white", fontsize=10, fontweight="bold", path_effects=_OUTLINE)
        self._dynamic_artists += [self.gg_trail, self.gg_ball, self.gg_txt]

    def _build_minimap(self) -> None:
        self.ax_map.set_aspect("equal")
        self.ax_map.plot(self.df["lon"], self.df["lat"], color="cyan", lw=2, alpha=0.35)
        self.map_dot, = self.ax_map.plot([], [], "o", color="white", mec="red", mew=2, ms=8)
        self.map_tail, = self.ax_map.plot([], [], color="#00ff9f", lw=3, alpha=0.9, path_effects=_OUTLINE)
        self._dynamic_artists += [self.map_dot, self.map_tail]

    def _build_session_graph(self) -> None:
        # A thin strip pinned along the bottom edge, not a full-height
        # panel -- there's no room for an axis title here, and the graph's
        # own contrasting color against the dim static trace already reads
        # as "this is the session telemetry" without one.
        self.ax_gph.plot(self.df["time"], self.df["speed"], color="white", alpha=0.25, lw=1)
        for i in self.lap_indices:
            self.ax_gph.axvline(self.df.iloc[i]["time"], color="yellow", ls="--", alpha=0.35)
        self.gph_line, = self.ax_gph.plot([], [], color="#00ff9f", lw=2.2, path_effects=_OUTLINE)
        self.gph_dot, = self.ax_gph.plot([], [], "o", color="#ff0055", ms=6, mec="white")
        self.ax_gph.set_xlim(self.df["time"].min(), self.df["time"].max())
        top = self.df["speed"].max()
        self.ax_gph.set_ylim(0, top * 1.1 if top > 0 else 1)
        self._dynamic_artists += [self.gph_line, self.gph_dot]

    def draw_frame(self, f: int) -> None:
        if f >= len(self.df):
            return
        row = self.df.iloc[f]
        cfg = self.config

        if cfg.enable_speedo:
            v = row["speed"]
            self.sp_txt.set_text(f"{int(v)}")
            r = min(v / cfg.max_expected_speed_kmh, 1.0)
            idx = int(r * 100)
            color = "#00ff9f" if r < 0.5 else "#ffff00" if r < 0.8 else "#ff0055"
            self.sp_arc.set_data(self._arc_x[:idx], self._arc_y[:idx])
            self.sp_arc.set_color(color)
            self.lap_txt.set_text(f"LAP {int(row.get('lap', 0))}")
            last_lap = row.get("last_lap_s", 0.0)
            self.last_txt.set_text(f"LAST: {last_lap:.2f}s" if last_lap > 0 else "")

        if cfg.enable_gg:
            hist = self.df.iloc[max(0, f - cfg.gg_trail_frames):f + 1]
            self.gg_trail.set_data(hist["lat_g"], hist["lon_g"])
            self.gg_ball.set_data([row["lat_g"]], [row["lon_g"]])
            g_val = float(np.sqrt(row["lat_g"] ** 2 + row["lon_g"] ** 2))
            self.gg_txt.set_text(f"{g_val:.2f} G")
            self.gg_ball.set_color("red" if g_val > 1.0 else "yellow" if g_val > 0.5 else "#00ff9f")

        if cfg.enable_minimap:
            self.map_dot.set_data([row["lon"]], [row["lat"]])
            mtail = self.df.iloc[max(0, f - cfg.minimap_trail_frames):f + 1]
            self.map_tail.set_data(mtail["lon"], mtail["lat"])

        if cfg.enable_session_graph:
            gph = self.df.iloc[:f + 1]
            self.gph_line.set_data(gph["time"], gph["speed"])
            self.gph_dot.set_data([row["time"]], [row["speed"]])

    def save_frame(self, path: Path) -> None:
        """Captures the current frame via blitting instead of `savefig` --
        see the module docstring for why this is the main render-speed
        lever. Pixel output is identical to a transparent `savefig` at the
        same DPI; `compress_level=1` only trades (irrelevant, temporary)
        file size for faster PNG encoding, not image quality -- PNG
        compression is lossless at every level."""
        self.fig.canvas.restore_region(self._background)
        for artist in self._dynamic_artists:
            artist.axes.draw_artist(artist)
        self.fig.canvas.blit(self.fig.bbox)

        width, height = self.fig.canvas.get_width_height()
        buf = self.fig.canvas.buffer_rgba()
        Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1).save(path, compress_level=1)

    def close(self) -> None:
        plt.close(self.fig)
