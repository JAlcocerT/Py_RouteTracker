import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import re
import os
import mplcyberpunk
import numpy as np

# --- 1. CONFIGURATION ---
# The User confirmed this is the correct file for the video
FILE_MAIN = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt" 
LABEL_MAIN = "Kart Run 1"

OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
TARGET_FPS = 18 
RENDER_SECONDS = 40 #532.5 # Full duration rendering by default for verification
RENDER_FULL = True #False     # Render all frames
SPEED_MULTIPLIER = 1.0 
MAX_EXPECTED_SPEED = 85 

# VIDEO_DURATION: The TOTAL duration of the GoPro video in seconds.
# We will stretch/compress all data points to fit this duration exactly.
# This answers: "how we will make sure it will work reliably in other videos?" -> By setting this one value.
VIDEO_DURATION_SEC = 532.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PARSING (BLIND INDEX MAPPING) ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        dd = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        if parts[3].strip() in ['S', 'W']: dd *= -1
        return dd
    except: return None

def parse_telemetry(file_path, video_duration_sec=VIDEO_DURATION_SEC):
    """
    Parses telemetry and linearly maps all points to the video duration.
    This ignores 'Sample Time' values completely and assumes a constant sample rate.
    """
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                
            lines = content.splitlines()
            data = []
            
            # Simple single-pass extraction
            current_lat, current_lon = np.nan, np.nan
            
            for line in lines:
                # Extract Speed
                m = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                if m:
                    speed = float(m.group(1))
                    data.append({
                        'speed_kmh': speed * SPEED_MULTIPLIER,
                        'lat': current_lat, 
                        'lon': current_lon
                    })
                    
                # Extract Location (update current state for subsequent points)
                lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                if lat_m: current_lat = dms_to_dd(lat_m.group(1))
                
                lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                if lon_m: current_lon = dms_to_dd(lon_m.group(1))
            
            if not data: continue

            df = pd.DataFrame(data)
            
            # Fill missing coordinates
            df['lat'] = df['lat'].ffill().bfill()
            df['lon'] = df['lon'].ffill().bfill()
            
            # Remove initial invalid locks
            df = df[(df['lat'] != 0) & (df['lon'] != 0)]
            
            num_points = len(df)
            if num_points < 2: return pd.DataFrame()
            
            # BLIND INDEX MAPPING: Map index 0..N to time 0..Duration
            df['time'] = np.linspace(0, video_duration_sec, num_points)
            df = df.set_index('time')
            
            # RESAMPLE to Target FPS
            full_time_range = np.arange(0, video_duration_sec, 1.0/TARGET_FPS)
            
            df_resampled = df.reindex(df.index.union(full_time_range)).interpolate(method='linear').reindex(full_time_range)
            df_resampled = df_resampled.reset_index().rename(columns={'index': 'time'})
            
            return df_resampled
            
        except Exception as e:
            continue
            
    return pd.DataFrame()

# --- 3. ROBUST HUD RENDER (V3b) ---
def render_racing_hud_v3b(df_data):
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 5), dpi=100) 
    gs = GridSpec(1, 4, figure=fig)
    
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2:])
    
    fig.patch.set_alpha(0.0)
    for ax in [ax_speed, ax_map, ax_graph]: ax.patch.set_alpha(0.0)
    ax_speed.axis('off'); ax_map.axis('off')

    # --- BACKGROUND PANELS ---
    bg_panel_1 = patches.Rectangle((-0.1, -0.1), 2.5, 1.2, transform=ax_speed.transAxes, 
                                   color='black', alpha=0.6, zorder=-10)
    ax_speed.add_patch(bg_panel_1)

    bg_panel_2 = patches.Rectangle((-0.05, -0.2), 1.1, 1.3, transform=ax_graph.transAxes, 
                                   color='black', alpha=0.6, zorder=-10)
    ax_graph.add_patch(bg_panel_2)

    # A. SPEEDOMETER
    theta = np.linspace(np.pi, 0, 100)
    arc_radius = 0.28
    arc_x = 0.5 + arc_radius * np.cos(theta)
    arc_y = 0.5 + arc_radius * np.sin(theta)
    
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.1)
    for deg in np.linspace(180, 0, 11):
        rad = np.deg2rad(deg)
        x_start, y_start = 0.5 + (arc_radius-0.03)*np.cos(rad), 0.5 + (arc_radius-0.03)*np.sin(rad)
        x_end, y_end = 0.5 + (arc_radius+0.02)*np.cos(rad), 0.5 + (arc_radius+0.02)*np.sin(rad)
        ax_speed.plot([x_start, x_end], [y_start, y_end], color='white', lw=1, alpha=0.4)

    speed_arc, = ax_speed.plot([], [], lw=6, alpha=0.9, solid_capstyle='round')
    peak_marker, = ax_speed.plot([], [], 'o', color='white', markersize=3, alpha=0.6)
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=35, color='white', ha='center', fontweight='bold')
    ax_speed.text(0.5, 0.38, 'KM/H', fontsize=9, color='#00ff9f', ha='center', alpha=0.8)
    ax_speed.set_xlim(0.15, 0.85); ax_speed.set_ylim(0.25, 0.85)

    # B. MAP
    ax_map.plot(df_data['lon'], df_data['lat'], color='cyan', lw=1, alpha=0.1)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=5)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=2, alpha=0.8)

    # C. GRAPH (Single Line)
    ax_graph.set_title(f"SPEED PROFILE: {LABEL_MAIN}", color='white', fontsize=10, loc='center', pad=10)
    
    # Plot Full Profile
    ax_graph.plot(df_data['time'], df_data['speed_kmh'], color='white', alpha=0.15, lw=1)
    
    # Plot Live Line
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=1.0)
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax_graph.set_xlim(0, df_data['time'].max())
    ax_graph.set_ylim(0, df_data['speed_kmh'].max() * 1.1)
    ax_graph.tick_params(colors='white', labelsize=8)
    for spine in ax_graph.spines.values(): 
        spine.set_edgecolor('white'); spine.set_visible(True)
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    ax_graph.grid(True, alpha=0.1, color='white', linestyle=':')

    plt.subplots_adjust(bottom=0.2)
    peak_speed = 0.0

    def update(frame):
        nonlocal peak_speed
        if frame < len(df_data):
            row = df_data.iloc[frame]
            val = row['speed_kmh']
            peak_speed = max(peak_speed, val)
            
            # Speedometer
            speed_text.set_text(f"{int(val)}")
            ratio = min(val / MAX_EXPECTED_SPEED, 1.0)
            fill_idx = int(ratio * 100)
            color = '#00ff9f' if ratio < 0.5 else '#ffff00' if ratio < 0.8 else '#ff0055'
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            speed_arc.set_color(color)
            
            peak_ratio = min(peak_speed / MAX_EXPECTED_SPEED, 1.0)
            peak_rad = np.deg2rad(180 - peak_ratio * 180)
            peak_marker.set_data([0.5 + (arc_radius+0.05)*np.cos(peak_rad)], [0.5 + (arc_radius+0.05)*np.sin(peak_rad)])

            # Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail = max(0, frame - 30)
            map_path.set_data(df_data['lon'][trail:frame+1], df_data['lat'][trail:frame+1])
            
            # Graph
            current_data = df_data.iloc[:frame+1]
            graph_line.set_data(current_data['time'], current_data['speed_kmh'])
            graph_ball.set_data([row['time']], [val])

        return speed_text, speed_arc, peak_marker, map_tracker, map_path, graph_line, graph_ball

    num_frames = len(df_data) if RENDER_FULL else int(RENDER_SECONDS * TARGET_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud_v3b.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=TARGET_FPS, bitrate=5000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error: {e}")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=TARGET_FPS)
    plt.close()
    return save_path

if __name__ == "__main__":
    print(f"Extracting Data from {FILE_MAIN}...")
    print(f"Target Video Duration: {VIDEO_DURATION_SEC}s")
    
    df = parse_telemetry(FILE_MAIN, VIDEO_DURATION_SEC)
    
    if not df.empty:
        print(f"Loaded {len(df)} frames for rendering.")
        # Optional: Quick check of data
        print(f"Initial Speed: {df.iloc[0]['speed_kmh']:.1f} km/h")
        print(f"Final Speed: {df.iloc[-1]['speed_kmh']:.1f} km/h")
        
        OUTPUT_FILE = render_racing_hud_v3b(df)
        print(f"\nSUCCESS! HUD v3b created in {OUTPUT_FILE}")
    else:
        print("Error: Could not load data file.")
