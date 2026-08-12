"""Synthetic 'video frame' background so HUD contrast can be judged against
something with real tonal variance, not flat black (which flatters any
overlay regardless of design)."""
from PIL import Image, ImageDraw, ImageFilter
import random
import math

W, H = 1600, 900
random.seed(7)

img = Image.new("RGB", (W, H))
px = img.load()

# sky-to-tarmac gradient (bright sky top, dark asphalt bottom) -- the
# trickiest case for HUD contrast, since a design that only works on dark
# backgrounds will wash out against the bright sky band
for y in range(H):
    t = y / H
    if t < 0.4:
        # bright overcast sky
        s = t / 0.4
        r = int(180 + 40 * s)
        g = int(195 + 30 * s)
        b = int(210 + 20 * s)
    else:
        # asphalt, darker toward the bottom
        s = (t - 0.4) / 0.6
        r = int(140 - 100 * s)
        g = int(140 - 100 * s)
        b = int(135 - 95 * s)
    for x in range(W):
        px[x, y] = (r, g, b)

img = img.filter(ImageFilter.GaussianBlur(2))
draw = ImageDraw.Draw(img)

# a light-colored barrier/kerb band across the lower third -- exactly the
# kind of bright, busy region a bottom-corner HUD has to sit on top of
draw.rectangle([0, 620, W, 700], fill=(225, 225, 210))
for i in range(0, W, 60):
    draw.rectangle([i, 620, i + 30, 700], fill=(200, 60, 60))

# a bright sun/glare blob upper-right -- worst case for a top-right widget
for r in range(220, 0, -4):
    a = int(60 * (1 - r / 220))
    draw.ellipse([1350 - r, 60 - r, 1350 + r, 60 + r], fill=(255, 250, 230))

# some trackside clutter (dark trees / stands) lower-left and lower-right
for cx, cy in [(120, 520), (1500, 500), (250, 470)]:
    draw.ellipse([cx - 90, cy - 60, cx + 90, cy + 60], fill=(40, 55, 35))

from pathlib import Path
img.save(Path(__file__).parent / "bg.png")
print("saved bg")
