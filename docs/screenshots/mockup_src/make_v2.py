"""V2 layout mockup: drop the central session-graph strip and the minimap,
enlarge the G-G diagram into the right corner, and replace the simple arc
gauge with a proper analog-style speedometer (tick marks, needle, digital
readout, colored redline zone). PNG-only iteration, no hud_layers.py
changes yet.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Wedge
from PIL import Image

W, H, DPI = 1600, 900, 100
OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]
THIN_OUTLINE = [pe.withStroke(linewidth=2, foreground="black")]
GREEN, YELLOW, RED = "#00ff9f", "#ffe14d", "#ff2d55"
WHITE = "#f4f6f8"

from pathlib import Path
BASE = str(Path(__file__).parent)
OUT_DIR = str(Path(__file__).parent.parent)

MAX_SPEED = 95.0
CUR_SPEED = 73.0
CUR_G = 0.98
G_LIMIT = 1.5

# gauge sweep: 225deg (bottom-left) -> -45deg (bottom-right), going through
# the top -- the standard automotive speedometer arc (270 degrees total)
GAUGE_START, GAUGE_END = 225.0, -45.0


def new_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_alpha(0.0)
    return fig


def panel(fig, rect, radius=0.02, alpha=0.42):
    l, b, w, h = rect
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.patch.set_alpha(0)
    box = FancyBboxPatch(
        (l, b), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor=(1, 1, 1, 0.18), facecolor=(0.02, 0.05, 0.06, alpha),
        transform=fig.transFigure, zorder=1,
    )
    ax.add_patch(box)
    return ax


def speed_to_angle(v: float) -> float:
    frac = np.clip(v / MAX_SPEED, 0, 1)
    return GAUGE_START + frac * (GAUGE_END - GAUGE_START)


def speedo_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0)

    # dial face -- a subtly darker disc than the surrounding panel, so the
    # instrument reads as its own object, plus a thin bezel ring
    ax.add_artist(Circle((0, 0), 1.0, facecolor=(0, 0, 0, 0.22), edgecolor=(1, 1, 1, 0.35), lw=1.6, zorder=1))
    ax.add_artist(Circle((0, 0), 1.0, facecolor="none", edgecolor=(1, 1, 1, 0.12), lw=6, zorder=1))

    # colored redline band just inside the tick ring -- subtle, not a bold fill
    band_thetas = np.radians(np.linspace(GAUGE_START, GAUGE_END, 200))
    band_r = 0.9
    for lo, hi, color in [(0.0, 0.55, GREEN), (0.55, 0.82, YELLOW), (0.82, 1.0, RED)]:
        seg = band_thetas[int(lo * len(band_thetas)):max(int(lo * len(band_thetas)) + 1, int(hi * len(band_thetas)))]
        if len(seg) < 2:
            continue
        ax.plot(band_r * np.cos(seg), band_r * np.sin(seg), color=color, lw=4, alpha=0.85, solid_capstyle="butt", zorder=2)

    # major/minor tick marks with numeric labels every other major tick
    major_step, minor_step = 20, 10
    v = 0.0
    while v <= MAX_SPEED + 0.01:
        ang = np.radians(speed_to_angle(v))
        is_major = abs((v / major_step) - round(v / major_step)) < 1e-6
        r0, r1 = (0.78, 0.98) if is_major else (0.85, 0.98)
        ax.plot([r0 * np.cos(ang), r1 * np.cos(ang)], [r0 * np.sin(ang), r1 * np.sin(ang)],
                 color=WHITE, lw=2.6 if is_major else 1.2, alpha=0.9 if is_major else 0.5, zorder=3)
        if is_major:
            lx, ly = 0.66 * np.cos(ang), 0.66 * np.sin(ang)
            ax.text(lx, ly, f"{int(v)}", color=WHITE, fontsize=11, ha="center", va="center",
                     fontweight="bold", path_effects=THIN_OUTLINE, zorder=3)
        v += minor_step

    # needle -- tapered polygon + center hub, classic analog look
    ang = np.radians(speed_to_angle(CUR_SPEED))
    perp = ang + np.pi / 2
    base_w = 0.045
    tip = (0.72 * np.cos(ang), 0.72 * np.sin(ang))
    base_l = (base_w * np.cos(perp), base_w * np.sin(perp))
    base_r = (-base_w * np.cos(perp), -base_w * np.sin(perp))
    tail = (-0.14 * np.cos(ang), -0.14 * np.sin(ang))
    needle = Polygon([tail, base_l, tip, base_r], closed=True, facecolor="#ff2d55", edgecolor="white", lw=1.0, zorder=5)
    ax.add_patch(needle)
    ax.add_artist(Circle((0, 0), 0.11, facecolor="#1a1f22", edgecolor="white", lw=1.6, zorder=6))
    ax.add_artist(Circle((0, 0), 0.035, facecolor=RED, edgecolor="none", zorder=7))

    # digital readout, lower-center of the dial (below the pivot, above the
    # bottom gap of the arc) -- analog + digital combo reads as more "modern
    # instrument cluster" than either alone
    ax.text(0, -0.42, f"{int(CUR_SPEED)}", color="white", fontsize=34, ha="center", va="center",
             fontweight="bold", path_effects=OUTLINE, zorder=6)
    ax.text(0, -0.60, "KM/H", color=GREEN, fontsize=10.5, ha="center", va="center",
             fontweight="bold", path_effects=THIN_OUTLINE, zorder=6)

    # lap / last-lap, tucked above the dial
    ax.text(-0.98, 1.05, "LAP 2", color="cyan", fontsize=14, ha="left", va="bottom", fontweight="bold", path_effects=THIN_OUTLINE, zorder=6)
    ax.text(0.98, 1.05, "LAST 41.82s", color=YELLOW, fontsize=12, ha="right", va="bottom", fontweight="bold", path_effects=THIN_OUTLINE, zorder=6)
    return ax


def gg_axes(fig, rect):
    ax = fig.add_axes(rect)
    lim = G_LIMIT * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0)

    ax.add_artist(Circle((0, 0), G_LIMIT, facecolor=(0, 0, 0, 0.22), edgecolor=(1, 1, 1, 0.30), lw=1.6, zorder=1))
    for g in (0.5, 1.0, 1.5):
        if g > G_LIMIT:
            continue
        ls = "-" if g == 1.5 else "--"
        ax.add_artist(Circle((0, 0), g, fill=False, edgecolor="white", alpha=0.28 if g < G_LIMIT else 0.45, lw=1.2, ls=ls, zorder=2))
        ax.text(0.06, g + 0.05, f"{g:g}G", color="white", fontsize=8.5, alpha=0.6, path_effects=THIN_OUTLINE, zorder=2)
    ax.axhline(0, color="white", alpha=0.15, lw=1, zorder=2)
    ax.axvline(0, color="white", alpha=0.15, lw=1, zorder=2)
    ax.text(0, -lim + 0.14, "BRAKE", color="white", fontsize=9, alpha=0.35, ha="center", zorder=2)
    ax.text(0, lim - 0.14, "ACCEL", color="white", fontsize=9, alpha=0.35, ha="center", zorder=2)
    ax.text(-lim + 0.12, 0, "LEFT", color="white", fontsize=9, alpha=0.35, va="center", rotation=90, zorder=2)
    ax.text(lim - 0.12, 0, "RIGHT", color="white", fontsize=9, alpha=0.35, va="center", rotation=-90, zorder=2)

    n = 40
    t = np.linspace(0, 1, n)
    trail_x = 0.35 * np.sin(t * 4) * (1 - t * 0.3)
    trail_y = 0.9 * np.cos(t * 3) * t
    ax.plot(trail_x, trail_y, color="cyan", lw=2.4, alpha=0.75, path_effects=THIN_OUTLINE, zorder=4)
    gx, gy = trail_x[-1], trail_y[-1]
    ball_color = RED if CUR_G > 1.0 else YELLOW if CUR_G > 0.5 else GREEN
    ax.plot([gx], [gy], "o", color=ball_color, mec="white", mew=1.8, ms=16, zorder=5)

    ax.text(-lim + 0.05, lim - 0.12, "G-FORCE", color="white", fontsize=12, fontweight="bold", ha="left", path_effects=THIN_OUTLINE, zorder=6)
    ax.text(-lim + 0.05, lim - 0.34, f"{CUR_G:.2f} G", color="white", fontsize=15, fontweight="bold", ha="left", path_effects=OUTLINE, zorder=6)
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


fig = new_fig()
LEFT_PANEL = (0.02, 0.05, 0.34, 0.52)
RIGHT_PANEL = (0.64, 0.05, 0.34, 0.52)
panel(fig, LEFT_PANEL)
panel(fig, RIGHT_PANEL)
speedo_axes(fig, [0.035, 0.065, 0.31, 0.50])
gg_axes(fig, [0.655, 0.065, 0.31, 0.50])
composite(fig, "hud-layout-v2-speedo-gg")
