import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import re
import os
import mplcyberpunk
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt" 
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
SAMPLE_FPS = 18
RENDER_SECONDS = 10 
RENDER_FULL = False 
SPEED_MULTIPLIER = 1.0 
MAX_EXPECTED_SPEED = 85 # For color scaling

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PARSING ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        dd = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        if parts[3].strip() in ['S', 'W']: dd *= -1
        return dd
    except: return None

def parse_telemetry(file_path):
    data = []
    current_sample_time = 0.0
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                if "Sample Time" in content:
                    lines = content.splitlines()
                    for line in lines:
                        m = re.search(r"Sample Time\s+:\s+([\d\.]+)\s+s", line)
                        if m: current_sample_time = float(m.group(1))
                        m = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                        if m:
                            speed = float(m.group(1))
                            data.append({
                                'time': current_sample_time + len([d for d in data if d['sample_time'] == current_sample_time])*(1.0/SAMPLE_FPS),
                                'sample_time': current_sample_time,
                                'speed_kmh': speed * SPEED_MULTIPLIER,
                                'lat': None, 'lon': None
                            })
                        if data:
                            lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                            if lat_m: data[-1]['lat'] = dms_to_dd(lat_m.group(1))
                            lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                            if lon_m: data[-1]['lon'] = dms_to_dd(lon_m.group(1))
                    df = pd.DataFrame(data)
                    # Forward fill to handle gaps, but then drop any rows that STILL have no coordinates
                    # (this happens at the very start if the GPS hasn't locked yet)
                    df['lat'] = df['lat'].ffill()
                    df['lon'] = df['lon'].ffill()
                    
                    # REMAIND: Drop rows without speed OR without valid GPS lock
                    df = df.dropna(subset=['speed_kmh', 'lat', 'lon'])
                    
                    # Avoid (0,0) jumps if they leaked through
                    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
                    
                    return df
        except: continue
    return pd.DataFrame()

# --- 3. ULTRA-COOL HUD V3 RENDER ---
def render_racing_hud_v3(df):
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 4), dpi=100)
    gs = GridSpec(1, 4, figure=fig)
    
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2:])
    
    fig.patch.set_alpha(0.0)
    for ax in [ax_speed, ax_map, ax_graph]: ax.patch.set_alpha(0.0)
    ax_speed.axis('off'); ax_map.axis('off')

    # A. ULTRA SPEEDOMETER
    theta = np.linspace(np.pi, 0, 100)
    arc_radius = 0.28 # Reduced radius
    arc_x = 0.5 + arc_radius * np.cos(theta)
    arc_y = 0.5 + arc_radius * np.sin(theta)
    
    # Static elements: Background & Ticks
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.1)
    # Radial Ticks
    for deg in np.linspace(180, 0, 11): # 10 intervals
        rad = np.deg2rad(deg)
        x_start, y_start = 0.5 + (arc_radius-0.03)*np.cos(rad), 0.5 + (arc_radius-0.03)*np.sin(rad)
        x_end, y_end = 0.5 + (arc_radius+0.02)*np.cos(rad), 0.5 + (arc_radius+0.02)*np.sin(rad)
        ax_speed.plot([x_start, x_end], [y_start, y_end], color='white', lw=1, alpha=0.4)

    # Dynamic elements
    speed_arc, = ax_speed.plot([], [], lw=6, alpha=0.9, solid_capstyle='round')
    peak_marker, = ax_speed.plot([], [], 'o', color='white', markersize=3, alpha=0.6)
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=35, color='white', ha='center', fontweight='bold')
    ax_speed.text(0.5, 0.38, 'KM/H', fontsize=9, color='#00ff9f', ha='center', alpha=0.8)
    ax_speed.set_xlim(0.15, 0.85); ax_speed.set_ylim(0.25, 0.85)

    # B. Compact Map
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=1, alpha=0.1)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=5)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=2, alpha=0.8)

    # C. Speed Graph
    ax_graph.set_title("SESSION TELEMETRY", color='white', fontsize=10, loc='center', pad=15)
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=1.0)
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=10, markeredgecolor='white', markeredgewidth=1)
    ax_graph.set_xlim(df['time'].min(), df['time'].max())
    ax_graph.set_ylim(0, df['speed_kmh'].max() * 1.1)
    ax_graph.tick_params(colors='white', labelsize=8)
    for spine in ax_graph.spines.values(): spine.set_color('#444444')

    peak_speed = 0.0

    def update(frame):
        nonlocal peak_speed
        if frame < len(df):
            row = df.iloc[frame]
            val = row['speed_kmh']
            peak_speed = max(peak_speed, val)
            
            # 1. Update Speed with color shifting
            speed_text.set_text(f"{int(val)}")
            ratio = min(val / MAX_EXPECTED_SPEED, 1.0)
            fill_idx = int(ratio * 100)
            
            # Color: Green -> Yellow -> Red
            color = '#00ff9f' if ratio < 0.5 else '#ffff00' if ratio < 0.8 else '#ff0055'
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            speed_arc.set_color(color)
            
            # Peak Speed dot
            peak_ratio = min(peak_speed / MAX_EXPECTED_SPEED, 1.0)
            peak_rad = np.deg2rad(180 - peak_ratio * 180)
            peak_marker.set_data([0.5 + (arc_radius+0.05)*np.cos(peak_rad)], [0.5 + (arc_radius+0.05)*np.sin(peak_rad)])
            
            # 2. Update Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail = max(0, frame - 30)
            map_path.set_data(df['lon'][trail:frame+1], df['lat'][trail:frame+1])
            
            # 3. Update Graph
            graph_line.set_data(df['time'][:frame+1], df['speed_kmh'][:frame+1])
            graph_ball.set_data([row['time']], [val])
            
        return speed_text, speed_arc, peak_marker, map_tracker, map_path, graph_line, graph_ball

    num_frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    mplcyberpunk.add_glow_effects(ax=ax_graph)
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud_v3.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=SAMPLE_FPS, bitrate=5000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error: {e}")
        save_path = save_path.replace(".mp4", ".gif")
        ani.save(save_path, writer='pillow', fps=SAMPLE_FPS)
    plt.close()
    return save_path

# --- RUN ---
if __name__ == "__main__":
    print("Extracting telemetry for HUD V3...")
    df_hud = parse_telemetry(INPUT_FILE)
    if not df_hud.empty:
        if not RENDER_FULL: print(f"Rendering HUD V3 mockup ({RENDER_SECONDS}s)...")
        render_racing_hud_v3(df_hud)
        print(f"\nSUCCESS! HUD V3 created in {OUTPUT_DIR}/racing_hud_v3.mp4")
    else:
        print("Could not parse data.")
