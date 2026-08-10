"""HUD drawing: ported from overlay/racing_hud_v7.py:252-338.

Two real changes from the original:
  1. Every widget (speed arc, G-G diagram, minimap, session graph) is
     independently toggle-able via RenderConfig instead of always-on.
  2. The figure is drawn with a transparent background (`transparent=True`
     on savefig, no `fig.patch.set_facecolor('black')`) and frames are
     saved as individual PNGs rather than encoded straight to video —
     video_render.py composites this PNG sequence onto the real footage
     via ffmpeg's `overlay` filter, closing the gap where v7 only ever
     produced a standalone HUD clip.

This module renders ONE frame at a time (`draw_frame`) rather than using
matplotlib.animation.FuncAnimation — FuncAnimation buys nothing once
video_render.py is doing the parallelization and file-writing itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

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


_OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]


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
        self._build_figure()

    def _build_figure(self) -> None:
        cfg = self.config
        style = "cyberpunk" if (cfg.theme == "cyberpunk" and _HAS_CYBERPUNK) else "dark_background"
        plt.style.use(style)

        fig_w, fig_h = cfg.width_px / cfg.dpi, cfg.height_px / cfg.dpi
        self.fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.dpi)
        self.fig.patch.set_alpha(0.0)

        gs = GridSpec(2, 3, height_ratios=[3, 1], figure=self.fig)
        self.ax_spd = self.fig.add_subplot(gs[0, 0])
        self.ax_gg = self.fig.add_subplot(gs[0, 1])
        self.ax_map = self.fig.add_subplot(gs[0, 2])
        self.ax_gph = self.fig.add_subplot(gs[1, :])

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
            mplcyberpunk.add_glow_effects(ax=self.ax_spd)

    def _build_speedo(self) -> None:
        theta = np.linspace(np.pi, 0, 100)
        rad = 0.35
        self._arc_x = 0.5 + rad * np.cos(theta)
        self._arc_y = 0.4 + rad * np.sin(theta)
        self.ax_spd.plot(self._arc_x, self._arc_y, color="white", lw=1, alpha=0.1)
        self.sp_arc, = self.ax_spd.plot([], [], lw=8, solid_capstyle="round", path_effects=_OUTLINE)
        self.sp_txt = self.ax_spd.text(0.5, 0.35, "", fontsize=45, color="white", ha="center", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.text(0.5, 0.25, "KM/H", fontsize=12, color="#00ff9f", ha="center", path_effects=_OUTLINE)
        self.lap_txt = self.ax_spd.text(0.1, 0.85, "LAP", fontsize=16, color="cyan", ha="left", fontweight="bold", path_effects=_OUTLINE)
        self.last_txt = self.ax_spd.text(0.9, 0.85, "LAST", fontsize=12, color="yellow", ha="right", fontweight="bold", path_effects=_OUTLINE)
        self.ax_spd.set_xlim(0, 1)
        self.ax_spd.set_ylim(0, 1)

    def _build_gg(self) -> None:
        cfg = self.config
        self.ax_gg.set_xlim(-cfg.limit_g, cfg.limit_g)
        self.ax_gg.set_ylim(-cfg.limit_g, cfg.limit_g)
        self.ax_gg.set_aspect("equal")
        self.ax_gg.add_artist(plt.Circle((0, 0), 0.5, color="white", fill=False, alpha=0.2, ls="--"))
        self.ax_gg.add_artist(plt.Circle((0, 0), 1.0, color="white", fill=False, alpha=0.4, ls="-"))
        self.ax_gg.axhline(0, color="white", alpha=0.1)
        self.ax_gg.axvline(0, color="white", alpha=0.1)
        self.gg_trail, = self.ax_gg.plot([], [], color="cyan", lw=2, alpha=0.6, path_effects=_OUTLINE)
        self.gg_ball, = self.ax_gg.plot([], [], "o", color="#ff0055", markersize=12, mec="white", zorder=10)
        self.gg_txt = self.ax_gg.text(0.05, 0.9, "", transform=self.ax_gg.transAxes, color="white", fontsize=10, path_effects=_OUTLINE)

    def _build_minimap(self) -> None:
        self.ax_map.set_aspect("equal")
        self.ax_map.plot(self.df["lon"], self.df["lat"], color="cyan", lw=2, alpha=0.3)
        self.map_dot, = self.ax_map.plot([], [], "o", color="white", mec="red", mew=2, ms=8)
        self.map_tail, = self.ax_map.plot([], [], color="#00ff9f", lw=3, alpha=0.9, path_effects=_OUTLINE)

    def _build_session_graph(self) -> None:
        self.ax_gph.set_title("SESSION TELEMETRY", color="white", fontsize=9, pad=5, path_effects=_OUTLINE)
        self.ax_gph.plot(self.df["time"], self.df["speed"], color="white", alpha=0.2, lw=1)
        for i in self.lap_indices:
            self.ax_gph.axvline(self.df.iloc[i]["time"], color="yellow", ls="--", alpha=0.3)
        self.gph_line, = self.ax_gph.plot([], [], color="#00ff9f", lw=2, path_effects=_OUTLINE)
        self.gph_dot, = self.ax_gph.plot([], [], "o", color="#ff0055", ms=6, mec="white")
        self.ax_gph.set_xlim(self.df["time"].min(), self.df["time"].max())
        top = self.df["speed"].max()
        self.ax_gph.set_ylim(0, top * 1.1 if top > 0 else 1)

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
        self.fig.savefig(path, transparent=True, dpi=self.config.dpi)

    def close(self) -> None:
        plt.close(self.fig)
