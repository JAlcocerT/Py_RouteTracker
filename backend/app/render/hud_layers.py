"""HUD drawing: ported from legacy/overlay/racing_hud_v7.py:252-338.

Four real changes from the original:
  1. Every widget (speedo, G-G diagram, minimap) is independently
     toggle-able via RenderConfig instead of always-on. The original's
     fourth widget, a session-wide speed graph running along the bottom
     edge, is gone entirely -- on real close-up POV footage (steering
     wheel/hands/knees filling the lower-center of frame) it had nowhere
     clear to sit and just added visual noise on top of the cockpit.
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
    max_expected_speed_kmh: float = 85.0
    limit_g: float = 1.5
    theme: str = "cyberpunk"  # "cyberpunk" or "dark_background"
    width_px: int = 1600
    height_px: int = 900
    dpi: int = 100
    gg_trail_frames: int = 15
    minimap_trail_frames: int = 150


def _format_lap_time(seconds: float) -> str:
    """m:ss.cc (minutes:seconds.centiseconds), the standard racing-timing
    format -- not the bare `ss.dd` a plain `f"{seconds:.2f}"` gives, which
    reads as "61.95" for a 1:01.95 lap instead of the familiar clock shape.
    Rounds via integer centiseconds so e.g. 59.998s correctly carries into
    "1:00.00" instead of rounding display digits independently and
    printing "0:60.00".
    """
    total_cs = int(round(seconds * 100))
    minutes, rem_cs = divmod(total_cs, 6000)
    secs, centis = divmod(rem_cs, 100)
    return f"{minutes}:{secs:02d}.{centis:02d}"


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

# Speedo sweep: 240 degrees of a full circle (a 120 degree gap centered on
# straight down, for the numeric readout) rather than a flat 180 degree
# semicircle -- closer to how a real gauge cluster reads. Angles in the
# standard math convention (0=3 o'clock, 90=12 o'clock, measured
# counter-clockwise), matching np.cos/np.sin directly.
_SWEEP_START_DEG = 210.0
_SWEEP_END_DEG = -30.0


def _angle_for_frac(frac: float) -> float:
    """Maps a 0..1 gauge reading to its angle (radians) along the sweep."""
    return np.radians(_SWEEP_START_DEG + frac * (_SWEEP_END_DEG - _SWEEP_START_DEG))

# Rounded, semi-transparent dark backing panels give every widget contrast
# against arbitrary, unpredictable footage (bright sky, glare, light-colored
# kerbs/barriers) that a bare glow/outline alone can't reliably beat -- see
# the mockups compared against a synthetic busy background before this
# layout was picked. Corner placement keeps the center of the frame, where
# the actual footage subject (and, on close-up POV shots, the cockpit)
# usually is, uncovered.
_PANEL_FACE_RGB = (0.02, 0.05, 0.06)
_PANEL_EDGE = (1, 1, 1, 0.18)

# Figure-fraction (left, bottom, width, height) / [left, bottom, width,
# height] rects. Kept as module constants rather than RenderConfig fields --
# this is a considered visual design, not a per-render knob callers should
# reasonably want to override (unlike the toggles/thresholds on
# RenderConfig itself).
#
# Speedo top-left, G-G and minimap both anchored along the bottom (left and
# right respectively). An earlier version kept the minimap top-right instead
# -- open track/sky in most helmet/chest-mount POV shots, versus the bottom
# corners' cockpit-occlusion risk (steering wheel, hands, knees) -- but this
# footage's bottom corners read as open track/ground, not cockpit, so both
# bottom slots are back in use. Reconsider (shrinking the minimap, or moving
# it back to a top corner) if it turns out to overlap the cockpit on other
# camera mounts/angles.
_SPEEDO_PANEL_RECT = (0.015, 0.56, 0.30, 0.40)
_GG_PANEL_RECT = (0.015, 0.035, 0.17, 0.21)
_MINIMAP_PANEL_RECT = (0.685, 0.035, 0.30, 0.40)
_SPEEDO_RECT = [0.02, 0.58, 0.29, 0.36]
_GG_RECT = [0.03, 0.05, 0.15, 0.19]
_MINIMAP_RECT = [0.70, 0.05, 0.28, 0.36]


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

        for ax, enabled in (
            (self.ax_spd, cfg.enable_speedo),
            (self.ax_gg, cfg.enable_gg),
            (self.ax_map, cfg.enable_minimap),
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
        if not (cfg.enable_speedo or cfg.enable_gg or cfg.enable_minimap):
            return

        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.patch.set_alpha(0)
        if cfg.enable_speedo:
            ax.add_patch(self._panel_patch(_SPEEDO_PANEL_RECT))
        if cfg.enable_gg:
            ax.add_patch(self._panel_patch(_GG_PANEL_RECT))
        if cfg.enable_minimap:
            ax.add_patch(self._panel_patch(_MINIMAP_PANEL_RECT))

    @staticmethod
    def _panel_patch(rect: tuple[float, float, float, float], radius: float = 0.02, alpha: float = 0.42) -> FancyBboxPatch:
        left, bottom, width, height = rect
        return FancyBboxPatch(
            (left, bottom), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=1.2, edgecolor=_PANEL_EDGE, facecolor=(*_PANEL_FACE_RGB, alpha),
        )

    def _build_speedo(self) -> None:
        """A real gauge -- redline zones and tick marks you can read a value
        against, plus a needle -- rather than the original's single
        recolouring arc, which read more like a loading bar than a
        speedometer since it had nothing static to compare the fill to."""
        cfg = self.config
        cx, cy, rad = 0.5, 0.38, 0.30
        self._spd_center = (cx, cy)
        self._spd_radius = rad
        theta = np.linspace(*[np.radians(d) for d in (_SWEEP_START_DEG, _SWEEP_END_DEG)], 100)
        self._arc_x = cx + rad * np.cos(theta)
        self._arc_y = cy + rad * np.sin(theta)

        # Static redline-zone track: green/yellow/red bands behind
        # everything else, so the needle has a fixed reference to read
        # against instead of just an abstract 0-1 fill.
        for lo, hi, color in (
            (0.0, 0.60, "#00ff9f"),
            (0.60, 0.85, "#ffd400"),
            (0.85, 1.0, "#ff0055"),
        ):
            zt = np.linspace(_angle_for_frac(lo), _angle_for_frac(hi), 24)
            self.ax_spd.plot(cx + rad * np.cos(zt), cy + rad * np.sin(zt),
                              color=color, lw=12, alpha=0.20, solid_capstyle="butt")

        # Static tick marks every 10% of the configured max speed, with a
        # numeric label on every other (major) tick.
        for frac in np.linspace(0, 1, 11):
            ang = _angle_for_frac(frac)
            major = round(frac * 10) % 2 == 0
            r0 = rad - (0.05 if major else 0.03)
            self.ax_spd.plot(
                [cx + r0 * np.cos(ang), cx + (rad + 0.012) * np.cos(ang)],
                [cy + r0 * np.sin(ang), cy + (rad + 0.012) * np.sin(ang)],
                color="white", lw=1.6 if major else 1.0, alpha=0.6 if major else 0.3,
            )
            if major:
                lx, ly = cx + (rad + 0.085) * np.cos(ang), cy + (rad + 0.085) * np.sin(ang)
                self.ax_spd.text(
                    lx, ly, f"{int(round(frac * cfg.max_expected_speed_kmh))}",
                    fontsize=8, color="white", alpha=0.65, ha="center", va="center",
                )

        # Dynamic accent arc (how far round we are, colour-matched to the
        # current zone) and the needle, the two moving indicators.
        self.sp_arc, = self.ax_spd.plot([], [], lw=4, solid_capstyle="round", alpha=0.9, path_effects=_OUTLINE)
        self.sp_needle, = self.ax_spd.plot([], [], color="white", lw=2.4, solid_capstyle="round", zorder=9, path_effects=_OUTLINE)
        self.ax_spd.add_patch(plt.Circle((cx, cy), 0.026, facecolor="#12141a", edgecolor="white", lw=1.3, zorder=10))

        self.sp_txt = self.ax_spd.text(cx, cy - 0.15, "", fontsize=32, color="white", ha="center", va="center", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.text(cx, cy - 0.235, "KM/H", fontsize=10, color="#00ff9f", ha="center", va="center", fontweight="bold", path_effects=_OUTLINE)

        self.lap_txt = self.ax_spd.text(0.03, 0.95, "", fontsize=14, color="cyan", ha="left", va="center", fontweight="bold", path_effects=_OUTLINE)
        self.last_txt = self.ax_spd.text(0.97, 0.95, "", fontsize=12, color="yellow", ha="right", va="center", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.set_xlim(0, 1)
        self.ax_spd.set_ylim(0, 1)
        self._dynamic_artists += [self.sp_arc, self.sp_needle, self.sp_txt, self.lap_txt, self.last_txt]

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
            color = "#00ff9f" if r < 0.60 else "#ffd400" if r < 0.85 else "#ff0055"
            self.sp_arc.set_data(self._arc_x[:idx], self._arc_y[:idx])
            self.sp_arc.set_color(color)

            cx, cy = self._spd_center
            ang = _angle_for_frac(r)
            tip = self._spd_radius - 0.055
            self.sp_needle.set_data([cx, cx + tip * np.cos(ang)], [cy, cy + tip * np.sin(ang)])
            self.sp_needle.set_color(color)

            self.lap_txt.set_text(f"LAP {int(row.get('lap', 0))}")
            last_lap = row.get("last_lap_s", 0.0)
            self.last_txt.set_text(f"LAST {_format_lap_time(last_lap)}" if last_lap > 0 else "")

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
