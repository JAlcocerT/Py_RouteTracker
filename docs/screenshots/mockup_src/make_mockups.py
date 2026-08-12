"""Three candidate HUD layouts, rendered as real matplotlib frames and
composited over a synthetic 'video' background (bright sky, kerb band,
glare, trackside clutter) so contrast/placement can actually be judged --
not just described.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

W, H, DPI = 1600, 900, 100
OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]
GREEN, YELLOW, RED = "#00ff9f", "#ffe14d", "#ff2d55"

from pathlib import Path
BASE = str(Path(__file__).parent)
OUT_DIR = str(Path(__file__).parent.parent)

# ---- synthetic session data (2 laps, ~85s each, go-kart-ish speed trace) ----
t = np.linspace(0, 170, 600)
speed = 55 + 30 * np.sin(t / 6) ** 2 + 5 * np.sin(t / 1.3)
speed = np.clip(speed, 8, 95)
lap_marks = [0, 85, 170]
angle = t / 170 * 4 * np.pi
lon = 0.002 * np.cos(angle) + np.linspace(0, 0.0005, len(t))
lat = 0.0015 * np.sin(angle * 1.3)
g_lat = 0.9 * np.sin(angle * 1.3) + 0.05 * np.random.default_rng(3).normal(size=len(t))
g_lon = 0.6 * np.cos(angle * 0.7) + 0.05 * np.random.default_rng(4).normal(size=len(t))

F = 420  # "current frame" index for the mockup
cur_speed = speed[F]
cur_g = float(np.hypot(g_lat[F], g_lon[F]))
speed_color = GREEN if cur_speed / 95 < 0.5 else YELLOW if cur_speed / 95 < 0.8 else RED


def new_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_alpha(0.0)
    return fig


def panel(fig, rect, radius=0.02, alpha=0.42):
    """A rounded, semi-transparent dark backing panel for contrast, in
    figure-fraction coordinates (l, b, w, h)."""
    l, b, w, h = rect
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0)
    box = FancyBboxPatch(
        (l, b), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor=(1, 1, 1, 0.18), facecolor=(0.02, 0.05, 0.06, alpha),
        transform=fig.transFigure, zorder=1,
    )
    ax.add_patch(box)
    return ax


def speedo_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.patch.set_alpha(0)
    theta = np.linspace(np.pi, 0, 100)
    rad = 0.36
    ax_x = 0.5 + rad * np.cos(theta)
    ax_y = 0.32 + rad * np.sin(theta)
    ax.plot(ax_x, ax_y, color="white", lw=2, alpha=0.15)
    r = min(cur_speed / 95, 1.0)
    idx = int(r * 100)
    ax.plot(ax_x[:idx], ax_y[:idx], lw=9, solid_capstyle="round", color=speed_color, path_effects=OUTLINE)
    ax.text(0.5, 0.30, f"{int(cur_speed)}", fontsize=48, color="white", ha="center", fontweight="bold", path_effects=OUTLINE)
    ax.text(0.5, 0.16, "KM/H", fontsize=13, color=GREEN, ha="center", fontweight="bold", path_effects=OUTLINE)
    ax.text(0.06, 0.90, "LAP 2", fontsize=16, color="cyan", ha="left", fontweight="bold", path_effects=OUTLINE)
    ax.text(0.94, 0.90, "LAST 41.82s", fontsize=13, color=YELLOW, ha="right", fontweight="bold", path_effects=OUTLINE)
    return ax


def minimap_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0)
    ax.plot(lon, lat, color="cyan", lw=2, alpha=0.35)
    tail = slice(max(0, F - 150), F + 1)
    ax.plot(lon[tail], lat[tail], color=GREEN, lw=3, alpha=0.95, path_effects=OUTLINE)
    ax.plot([lon[F]], [lat[F]], "o", color="white", mec="red", mew=2, ms=9, zorder=5)
    return ax


def gg_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0)
    ax.add_artist(Circle((0, 0), 0.75, color="white", fill=False, alpha=0.25, ls="--"))
    ax.add_artist(Circle((0, 0), 1.5, color="white", fill=False, alpha=0.4))
    tail = slice(max(0, F - 15), F + 1)
    ax.plot(g_lat[tail], g_lon[tail], color="cyan", lw=2, alpha=0.7, path_effects=OUTLINE)
    ball_color = RED if cur_g > 1.0 else YELLOW if cur_g > 0.5 else GREEN
    ax.plot([g_lat[F]], [g_lon[F]], "o", color=ball_color, mec="white", mew=1.5, ms=13, zorder=5)
    ax.text(0.04, 0.90, f"{cur_g:.2f} G", transform=ax.transAxes, color="white", fontsize=12, fontweight="bold", path_effects=OUTLINE)
    return ax


def session_graph_axes(fig, rect, compact=False):
    ax = fig.add_axes(rect)
    ax.axis("off"); ax.patch.set_alpha(0)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, speed.max() * 1.15)
    ax.plot(t, speed, color="white", alpha=0.25, lw=1)
    for lm in lap_marks[1:-1]:
        ax.axvline(lm, color=YELLOW, ls="--", alpha=0.35, lw=1)
    ax.plot(t[:F + 1], speed[:F + 1], color=GREEN, lw=2.2, path_effects=OUTLINE)
    ax.plot([t[F]], [speed[F]], "o", color=RED, ms=7, mec="white", zorder=5)
    if not compact:
        ax.text(0.01, 0.82, "SESSION", transform=ax.transAxes, color="white", fontsize=9, fontweight="bold", path_effects=OUTLINE)
    return ax


def composite(fig, name):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    hud = Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).copy()
    bg = Image.open(f"{BASE}/bg.png").convert("RGBA")
    out = Image.alpha_composite(bg, hud)
    out.convert("RGB").save(f"{OUT_DIR}/{name}.png", quality=92)
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------------------
# Design A: corner panels -- rounded translucent backing behind each widget,
# tucked into the bottom-left / bottom-right corners; GG folded into the
# bottom-left panel under the speedo; thin full-width session strip pinned
# along the very bottom edge, behind/between the two corner panels.
# ---------------------------------------------------------------------------
figA = new_fig()
panel(figA, (0.015, 0.03, 0.30, 0.40))
panel(figA, (0.685, 0.03, 0.30, 0.40))
panel(figA, (0.015, 0.445, 0.97, 0.09), radius=0.015, alpha=0.35)
speedo_axes(figA, [0.02, 0.20, 0.29, 0.27])
gg_axes(figA, [0.045, 0.045, 0.10, 0.14])
minimap_axes(figA, [0.71, 0.06, 0.25, 0.36])
session_graph_axes(figA, [0.03, 0.455, 0.94, 0.07], compact=True)
composite(figA, "design_A_corner_panels")

# ---------------------------------------------------------------------------
# Design B: one continuous bottom bar -- speed+lap on the left third, a
# wider session sparkline in the middle, minimap on the right third; GG
# moved up to a small unobtrusive top-right gauge since it's the least
# glanceable-at-a-distance widget.
# ---------------------------------------------------------------------------
figB = new_fig()
panel(figB, (0.0, 0.0, 1.0, 0.30), radius=0.0, alpha=0.5)
speedo_axes(figB, [0.01, 0.015, 0.20, 0.275])
session_graph_axes(figB, [0.235, 0.05, 0.45, 0.20])
minimap_axes(figB, [0.70, 0.02, 0.27, 0.27])
panel(figB, (0.855, 0.70, 0.13, 0.22))
gg_axes(figB, [0.865, 0.715, 0.11, 0.19])
composite(figB, "design_B_bottom_bar")

# ---------------------------------------------------------------------------
# Design C: floating glow, no background panels at all -- relies purely on
# heavy black outline/shadow strokes for contrast against a busy
# background; bottom-left speed, bottom-right minimap+GG combined, a thin
# full-width sparkline glued to the very bottom edge with a soft drop
# shadow band beneath it.
# ---------------------------------------------------------------------------
figC = new_fig()
ax_shadow = figC.add_axes([0, 0, 1, 1]); ax_shadow.axis("off"); ax_shadow.patch.set_alpha(0)
ax_shadow.set_xlim(0, 1); ax_shadow.set_ylim(0, 1)
from matplotlib.patches import Rectangle
ax_shadow.add_patch(Rectangle((0, 0.0), 1, 0.05, transform=figC.transFigure, facecolor=(0, 0, 0, 0.55), edgecolor="none"))
speedo_axes(figC, [0.01, 0.20, 0.27, 0.30])
minimap_axes(figC, [0.735, 0.16, 0.24, 0.34])
gg_axes(figC, [0.775, 0.02, 0.16, 0.16])
session_graph_axes(figC, [0.02, 0.005, 0.96, 0.055], compact=True)
composite(figC, "design_C_floating_glow")
