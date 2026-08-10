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
RENDER_SECONDS = 20
RENDER_FULL = False 
SPEED_MULTIPLIER = 1.0 # kart data seems to be in km/h already in this file? or m/s?

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PARSING ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        direction = parts[3].strip()
        dd = degrees + minutes/60 + seconds/3600
        if direction in ['S', 'W']:
            dd *= -1
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
                            micro_offset = len([d for d in data if d['sample_time'] == current_sample_time]) * (1.0/SAMPLE_FPS)
                            data.append({
                                'time': current_sample_time + micro_offset,
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
                    df['lat'] = df['lat'].ffill()
                    df['lon'] = df['lon'].ffill()
                    return df.dropna(subset=['speed_kmh'])
        except: continue
    return pd.DataFrame()

# --- 3. PREMIUM HUD V2B RENDER ---
def render_racing_hud_v2b(df):
    plt.style.use('cyberpunk')
    
    # GridSpec for asymmetrical layout: [Speed (1)] [Map (1)] [Graph (2)]
    fig = plt.figure(figsize=(16, 4), dpi=100)
    gs = GridSpec(1, 4, figure=fig)
    
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2:]) # Graph takes 2 columns
    
    fig.patch.set_alpha(0.0)
    for ax in [ax_speed, ax_map, ax_graph]:
        ax.patch.set_alpha(0.0)
        
    ax_speed.axis('off')
    ax_map.axis('off')

    # A. Compact Speedometer
    theta = np.linspace(np.pi, 0, 100)
    arc_radius = 0.3
    arc_x = 0.5 + arc_radius * np.cos(theta)
    arc_y = 0.5 + arc_radius * np.sin(theta)
    
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.3)
    speed_arc, = ax_speed.plot([], [], color='#00ff9f', lw=5, alpha=0.9)
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=40, color='white', ha='center', fontweight='bold')
    ax_speed.text(0.5, 0.38, 'KM/H', fontsize=10, color='#00ff9f', ha='center')
    ax_speed.set_xlim(0.1, 0.9)
    ax_speed.set_ylim(0.2, 0.8)

    # B. Compact Map
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=1, alpha=0.1)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=4)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=1.5, alpha=0.7)

    # C. Detailed Speed Graph (Centered Line & Tracker)
    ax_graph.set_title("SPEED TELEMETRY", color='white', fontsize=10, loc='center', pad=10)
    # Highlighted line up to current frame
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=1.0)
    # The moving "red ball"
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    ax_graph.set_xlim(df['time'].min(), df['time'].max())
    ax_graph.set_ylim(0, df['speed_kmh'].max() * 1.1)
    ax_graph.tick_params(colors='white', labelsize=8)
    for spine in ax_graph.spines.values():
        spine.set_color('#444444')
        if spine in [ax_graph.spines['top'], ax_graph.spines['right']]:
            spine.set_visible(False)

    def update(frame):
        if frame < len(df):
            row = df.iloc[frame]
            val = row['speed_kmh']
            curr_t = row['time']
            
            # Update Speed
            speed_text.set_text(f"{int(val)}")
            arc_fill = int(min(val / 60.0, 1.0) * 100)
            speed_arc.set_data(arc_x[:arc_fill], arc_y[:arc_fill])
            
            # Update Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail = max(0, frame - 30)
            map_path.set_data(df['lon'][trail:frame+1], df['lat'][trail:frame+1])
            
            # Update Graph
            graph_line.set_data(df['time'][:frame+1], df['speed_kmh'][:frame+1])
            graph_ball.set_data([curr_t], [val])
            
        return speed_text, speed_arc, map_tracker, map_path, graph_line, graph_ball

    num_frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    mplcyberpunk.add_glow_effects(ax=ax_graph)
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud_v2b.mp4")
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
print("Extracting telemetry for HUD v2b...")
df_hud = parse_telemetry(INPUT_FILE)
if not df_hud.empty:
    if not RENDER_FULL:
        print(f"Rendering HUD v2b mockup ({RENDER_SECONDS}s)...")
    render_racing_hud_v2b(df_hud)
    print(f"\nSUCCESS! HUD v2b created in {OUTPUT_DIR}/racing_hud_v2b.mp4")
else:
    print("Could not parse data.")
