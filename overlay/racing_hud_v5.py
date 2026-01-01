import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import re
import os
import mplcyberpunk
import numpy as np
import subprocess

# --- 1. CONFIGURATION ---
# The only thing you need to change:
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4"

OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
TARGET_FPS = 18 
RENDER_SECONDS = 30
RENDER_FULL = True
MAX_EXPECTED_SPEED = 85 

# Lap Logic Config (From lap_timer_v4a)
LAP_START_TIME_SEC = 5.0 
LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. AUTOMATION HELPERS ---
def extract_telemetry(video_path):
    """
    Runs exiftool -ee on the video to generate a .txt file.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    txt_path = os.path.join(video_dir, f"{base_name}_telemetry.txt")
    
    if os.path.exists(txt_path):
        print(f"Telemetry file found: {txt_path}")
        return txt_path
        
    print(f"Extracting telemetry from {video_path}...")
    try:
        with open(txt_path, "w") as outfile:
            subprocess.run(["exiftool", "-ee", video_path], stdout=outfile, check=True)
        print("Extraction complete.")
        return txt_path
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def get_video_duration(video_path):
    """Get the duration of a video file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}. Using fallback.")
        return 532.5 

# --- 3. EXECUTE AUTOMATION ---
FILE_MAIN = extract_telemetry(VIDEO_PATH)
if not FILE_MAIN: exit(1)

VIDEO_DURATION_SEC = get_video_duration(VIDEO_PATH)
print(f"Detected Duration: {VIDEO_DURATION_SEC:.2f}s")

# --- 4. DATA & LAP PARSING ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        dd = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        if parts[3].strip() in ['S', 'W']: dd *= -1
        return dd
    except: return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_coordinates_at_time(df, target_time):
    idx = (df['time'] - target_time).abs().idxmin()
    row = df.iloc[idx]
    return row['lat'], row['lon']

def parse_telemetry_with_laps(file_path):
    print(f"Reading {file_path}...")
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    df = pd.DataFrame()
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            lines = content.splitlines()
            data = []
            cur_lat, cur_lon = np.nan, np.nan
            
            for line in lines:
                m = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                if m:
                    speed = float(m.group(1)) * 3.6
                    data.append({
                        'speed_kmh': speed,
                        'lat': cur_lat, 'lon': cur_lon
                    })
                lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                if lat_m: cur_lat = dms_to_dd(lat_m.group(1))
                lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                if lon_m: cur_lon = dms_to_dd(lon_m.group(1))
            
            if not data: continue
            df = pd.DataFrame(data)
            df['lat'] = df['lat'].ffill().bfill()
            df['lon'] = df['lon'].ffill().bfill()
            df = df[(df['lat'] != 0) & (df['lon'] != 0)]
            if len(df) < 2: return pd.DataFrame(), []
            
            # Resample
            df['time'] = np.linspace(0, VIDEO_DURATION_SEC, len(df))
            df = df.set_index('time')
            full_time_range = np.arange(0, VIDEO_DURATION_SEC, 1.0/TARGET_FPS)
            df_re = df.reindex(df.index.union(full_time_range)).interpolate(method='linear').reindex(full_time_range)
            df = df_re.reset_index().rename(columns={'index': 'time'})
            break
        except: continue
        
    if df.empty: return df, []

    # Lap Detection
    print("Detecting Laps...")
    start_lat, start_lon = get_coordinates_at_time(df, LAP_START_TIME_SEC)
    
    lap_indices = []
    last_crossing_time = -MIN_LAP_TIME_SEC
    in_zone = False
    closest_dist = 99999.0
    best_idx = -1
    
    for i, row in df.iterrows():
        dist = haversine_distance(row['lat'], row['lon'], start_lat, start_lon)
        if (row['time'] - last_crossing_time) > MIN_LAP_TIME_SEC:
            if dist < LAP_DETECTION_RADIUS_M:
                if not in_zone:
                    in_zone = True
                    closest_dist = dist
                    best_idx = i
                else:
                    if dist < closest_dist:
                        closest_dist = dist
                        best_idx = i
            else:
                if in_zone:
                    in_zone = False
                    lap_indices.append(best_idx)
                    last_crossing_time = df.iloc[best_idx]['time']
    
    df['lap_number'] = 0
    df['last_lap_time'] = 0.0
    
    prev_idx = 0
    current_lap = 0
    for i, idx in enumerate(lap_indices):
        current_lap = i + 1
        df.loc[prev_idx:idx, 'lap_number'] = current_lap
        if i > 0:
            t_prev = df.iloc[lap_indices[i-1]]['time']
            t_curr = df.iloc[idx]['time']
            df.loc[idx:, 'last_lap_time'] = t_curr - t_prev
        prev_idx = idx
    df.loc[prev_idx:, 'lap_number'] = current_lap + 1
    
    return df, lap_indices

# --- 5. RENDERER ---
def render_racing_hud_v5(df_data, lap_indices):
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 5), dpi=100) 
    gs = GridSpec(1, 4, figure=fig)
    
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2:])
    
    fig.patch.set_facecolor('black')
    for ax in [ax_speed, ax_map, ax_graph]: ax.set_facecolor('black')
    ax_speed.axis('off'); ax_map.axis('off')
    outline = [pe.withStroke(linewidth=3, foreground='black')]

    # Speed
    theta = np.linspace(np.pi, 0, 100)
    arc_radius = 0.28
    arc_x = 0.5 + arc_radius * np.cos(theta)
    arc_y = 0.5 + arc_radius * np.sin(theta)
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.1)
    speed_arc, = ax_speed.plot([], [], lw=6, alpha=0.9, solid_capstyle='round', path_effects=[pe.withStroke(linewidth=4, foreground='black')])
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=35, color='white', ha='center', fontweight='bold', path_effects=outline)
    ax_speed.text(0.5, 0.38, 'KM/H', fontsize=9, color='#00ff9f', ha='center', alpha=0.9, path_effects=outline)
    lap_text = ax_speed.text(0.1, 0.8, 'LAP -', fontsize=14, color='cyan', ha='left', fontweight='bold', path_effects=outline)
    last_lap_text = ax_speed.text(0.9, 0.8, 'LAST: --.--', fontsize=10, color='yellow', ha='right', fontweight='bold', path_effects=outline)
    ax_speed.set_xlim(0.05, 0.95); ax_speed.set_ylim(0.25, 0.90)

    # Map
    ax_map.set_aspect('equal', adjustable='datalim')
    ax_map.plot(df_data['lon'], df_data['lat'], color='cyan', lw=1.5, alpha=0.3)
    map_tracker, = ax_map.plot([], [], 'o', color='white', markeredgecolor='red', markeredgewidth=1.5, markersize=6, zorder=5)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=2.5, alpha=0.9)
    map_path.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # Graph
    ax_graph.set_title(f"SPEED PROFILE", color='white', fontsize=10, pad=10, path_effects=outline)
    ax_graph.plot(df_data['time'], df_data['speed_kmh'], color='white', alpha=0.2, lw=1)
    for idx in lap_indices:
        t = df_data.iloc[idx]['time']
        ax_graph.axvline(x=t, color='yellow', linestyle='--', alpha=0.3, lw=1)
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=1.0, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=8, markeredgecolor='white')
    ax_graph.set_xlim(0, df_data['time'].max())
    ax_graph.set_ylim(0, df_data['speed_kmh'].max() * 1.1)
    for spine in ax_graph.spines.values(): spine.set_edgecolor('white'); spine.set_path_effects(outline)
    ax_graph.spines['top'].set_visible(False); ax_graph.spines['right'].set_visible(False)
    ax_graph.grid(True, alpha=0.15, color='white', linestyle=':')
    ax_graph.tick_params(colors='white', labelsize=8)
    for label in ax_graph.get_xticklabels() + ax_graph.get_yticklabels(): label.set_path_effects(outline)

    # Animation
    plt.subplots_adjust(bottom=0.2)
    def update(frame):
        if frame < len(df_data):
            row = df_data.iloc[frame]
            val = row['speed_kmh']
            
            speed_text.set_text(f"{int(val)}")
            ratio = min(val / MAX_EXPECTED_SPEED, 1.0)
            fill_idx = int(ratio * 100)
            color = '#00ff9f' if ratio < 0.5 else '#ffff00' if ratio < 0.8 else '#ff0055'
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            speed_arc.set_color(color)
            
            cur_lap = int(row['lap_number'])
            last_time = row['last_lap_time']
            lap_text.set_text(f"LAP {cur_lap}")
            if last_time > 0: last_lap_text.set_text(f"LAST: {last_time:.2f}s")
            else: last_lap_text.set_text("") 

            map_tracker.set_data([row['lon']], [row['lat']])
            trail = max(0, frame - 100)
            map_path.set_data(df_data['lon'][trail:frame+1], df_data['lat'][trail:frame+1])
            
            cur_data = df_data.iloc[:frame+1]
            graph_line.set_data(cur_data['time'], cur_data['speed_kmh'])
            graph_ball.set_data([row['time']], [val])

        return speed_text, speed_arc, lap_text, last_lap_text, map_tracker, map_path, graph_line, graph_ball

    num_frames = len(df_data) if RENDER_FULL else int(RENDER_SECONDS * TARGET_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    
    # Save with unique name based on input video
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    save_path = os.path.join(OUTPUT_DIR, f"HUD_v5_{base_name}.mp4")
    
    print(f"Rendering {num_frames} frames to {save_path}...")
    ani.save(save_path, writer='ffmpeg', fps=TARGET_FPS, bitrate=5000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()
    return save_path

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        exit(1)

    df, laps = parse_telemetry_with_laps(FILE_MAIN)
    if not df.empty:
        print(f"Loaded {len(df)} frames. Found {len(laps)} Laps.")
        OUTPUT_FILE = render_racing_hud_v5(df, laps)
        print(f"\nSUCCESS! HUD v5 created: {OUTPUT_FILE}")
    else:
        print("Error: Could not load data.")
