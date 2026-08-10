import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import os
import subprocess
import struct
import mplcyberpunk

# --- CONFIGURATION ---
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
FPS = 18 # Target FPS for overlay
RENDER_SECONDS = 30
RENDER_FULL = True

# Calibration (Auto-Calculated later)
ONE_G_RAW = 5014.0 # Default fallback

# Axis Mapping (Based on mean checks)
# C1 = Vertical (Gravity) ~ 1G
# C2 = Lateral (X)
# C3 = Longitudinal (Z)
AXIS_LAT = 'c2'
AXIS_LON = 'c3'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. EXTRACTION LOGIC (Embedded) ---
def extract_and_parse_gpmd(video_path):
    print(f"Extracting G-Force from {video_path}...")
    bin_path = video_path.replace(".MP4", ".bin").replace(".mp4", ".bin")
    
    # Dump
    if not os.path.exists(bin_path):
        try:
            # Find stream
            cmd_probe = ["ffprobe", "-v", "error", "-select_streams", "d", "-show_entries", "stream=index:stream_tags=handler_name", "-of", "csv=p=0", video_path]
            result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, text=True)
            stream_index = 3 # Default
            for line in result.stdout.strip().splitlines():
                if "GoPro MET" in line: stream_index = int(line.split(',')[0]); break
            
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-map", f"0:{stream_index}", "-f", "data", "-c", "copy", bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

    if not os.path.exists(bin_path): return pd.DataFrame()

    # Parse
    print("Parsing binary...")
    with open(bin_path, "rb") as f: content = f.read()
    
    data_points = []
    i = 0; length = len(content)
    while i < length - 8:
        key = content[i:i+4].decode('ascii', errors='ignore')
        if key == "ACCL":
            try:
                # content[i+4] is type, i+5 size, i+6 repeat
                elem_size = content[i+5]
                repeat = struct.unpack(">H", content[i+6:i+8])[0]
                payload_start = i + 8
                total_bytes = elem_size * repeat
                padded = (total_bytes + 3) & ~3
                
                for k in range(repeat):
                    off = payload_start + (k * elem_size)
                    # Assuming Signed Short (standard)
                    val = struct.unpack(">hhh", content[off:off+6])
                    data_points.append(val)
                i += 8 + padded
                continue
            except: pass
        i += 1
        
    df = pd.DataFrame(data_points, columns=['c1', 'c2', 'c3'])
    return df

def get_video_duration(video_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        return float(subprocess.check_output(cmd).decode().strip())
    except: return 532.5

# --- 2. PREP DATA ---
df = extract_and_parse_gpmd(VIDEO_PATH)
if df.empty:
    print("No data extracted.")
    exit(1)

# AUTO-CALIBRATION
# Assumption: Median Magnitude over the session is exactly 1G (Gravity)
# This works because the kart spends most time "flat" or balancing forces.
mag = np.sqrt(df['c1']**2 + df['c2']**2 + df['c3']**2)
ONE_G_RAW = mag.median()
print(f"Auto-Calibrated 1G Raw Value: {ONE_G_RAW:.2f}")

# Axis Detection (Optional: Check which axis is ~1G)
means = df[['c1', 'c2', 'c3']].abs().mean()
print(f"Axis Means: C1={means['c1']:.0f}, C2={means['c2']:.0f}, C3={means['c3']:.0f}")
# If C1 is dominant linear mean -> Vertical

# Normalize to Gs
df['lat_g'] = df[AXIS_LAT] / ONE_G_RAW
df['lon_g'] = df[AXIS_LON] / ONE_G_RAW

# Time Mapping
# We assume linear distribution over video duration
duration = get_video_duration(VIDEO_PATH)
total_samples = len(df)
df['time'] = np.linspace(0, duration, total_samples)
print(f"Data: {total_samples} samples over {duration:.1f}s (~{total_samples/duration:.0f} Hz)")

# Smooth data for visualization (Raw 200Hz is noisy)
# Rolling mean over ~0.1s (20 samples)
df['lat_g'] = df['lat_g'].rolling(window=20, center=True).mean().fillna(0)
df['lon_g'] = df['lon_g'].rolling(window=20, center=True).mean().fillna(0)

# Resample to Overlay FPS
t_target = np.arange(0, duration, 1/FPS)
df = df.set_index('time')
df_resampled = df.reindex(df.index.union(t_target)).interpolate(method='linear').reindex(t_target).reset_index()
df = df_resampled.rename(columns={'index': 'time'})

# --- 3. RENDER ---
print(f"Rendering G-Force HUD... ({len(df)} frames)")

plt.style.use("cyberpunk")
# Square figure for G-Circle
fig = plt.figure(figsize=(5, 5), dpi=100)
# Make transparent bg
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111)
ax.set_facecolor('black')
ax.axis('off')

# Config
LIMIT_G = 1.5 
ax.set_xlim(-LIMIT_G, LIMIT_G)
ax.set_ylim(-LIMIT_G, LIMIT_G)
ax.set_aspect('equal')

# Static Elements (Circles)
outline = [pe.withStroke(linewidth=3, foreground='black')]
# 0.5G
circle_05 = plt.Circle((0, 0), 0.5, color='white', fill=False, alpha=0.3, ls='--', lw=1)
ax.add_artist(circle_05)
ax.text(0, 0.52, '0.5G', color='white', fontsize=8, ha='center', alpha=0.5, path_effects=outline)

# 1.0G
circle_10 = plt.Circle((0, 0), 1.0, color='white', fill=False, alpha=0.5, ls='-', lw=1.5)
ax.add_artist(circle_10)
ax.text(0, 1.02, '1.0G', color='white', fontsize=8, ha='center', alpha=0.8, path_effects=outline)

# 1.5G
circle_15 = plt.Circle((0, 0), 1.5, color='red', fill=False, alpha=0.2, ls=':', lw=1)
ax.add_artist(circle_15)

# Crosshairs
ax.axhline(0, color='white', alpha=0.1, lw=1)
ax.axvline(0, color='white', alpha=0.1, lw=1)

# Dynamic Elements
# Trail (last N frames)
TRAIL_LEN = 15
trail_line, = ax.plot([], [], color='cyan', alpha=0.6, lw=2, path_effects=outline)

# Current Ball
ball, = ax.plot([], [], 'o', color='#ff0055', markersize=12, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
ball.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

# Text
g_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', path_effects=outline)

def update(frame):
    if frame >= len(df): return ball,
    
    row = df.iloc[frame]
    lat = row['lat_g']
    lon = row['lon_g']
    
    # Update Ball
    ball.set_data([lat], [lon])
    
    # Update Trail
    start = max(0, frame - TRAIL_LEN)
    history = df.iloc[start:frame+1]
    trail_line.set_data(history['lat_g'], history['lon_g'])
    
    # Total G
    total_g = np.sqrt(lat**2 + lon**2)
    g_text.set_text(f"{total_g:.2f} G")
    
    # Dynamic Color based on G
    if total_g > 1.0: ball.set_color('red')
    elif total_g > 0.5: ball.set_color('yellow')
    else: ball.set_color('#00ff9f') # Green/Teal

    return ball, trail_line, g_text

frames_to_render = len(df) if RENDER_FULL else int(RENDER_SECONDS * FPS)
ani = FuncAnimation(fig, update, frames=frames_to_render, blit=True)

# Add Glow
mplcyberpunk.add_glow_effects(ax=ax)

save_path = os.path.join(OUTPUT_DIR, "GForce_HUD_v1.mp4")
ani.save(save_path, writer='ffmpeg', fps=FPS, bitrate=3000, 
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']) # yuv420p for compatibility

print(f"Saved: {save_path}")
