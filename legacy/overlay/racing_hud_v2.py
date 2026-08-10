import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
import os
import mplcyberpunk
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt" 
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
SAMPLE_FPS = 18
RENDER_SECONDS = 10 
RENDER_FULL = True 
SPEED_MULTIPLIER = 1.0 # Set to 3.6 for km/h if raw data is in m/s

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
    except:
        return None

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

# --- 3. PREMIUM HUD V2 RENDER ---
def render_racing_hud_v2(df):
    plt.style.use('cyberpunk')
    
    # 1. Layout: 3 panels (Speed, Map, Graph)
    # We use a wider figure for the 3 components
    fig, (ax_speed, ax_map, ax_graph) = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    fig.patch.set_alpha(0.0)
    
    for ax in [ax_speed, ax_map, ax_graph]:
        ax.patch.set_alpha(0.0)
        
    ax_speed.axis('off')
    ax_map.axis('off')

    # A. Speedometer (Smaller than v1)
    theta = np.linspace(np.pi, 0, 100)
    arc_x = 0.5 + 0.35 * np.cos(theta) # Reduced radius
    arc_y = 0.5 + 0.35 * np.sin(theta)
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.3)
    speed_arc, = ax_speed.plot([], [], color='#00ff9f', lw=6, alpha=0.9)
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=50, color='white', ha='center', fontweight='bold')
    ax_speed.text(0.5, 0.35, 'KM/H', fontsize=12, color='#00ff9f', ha='center')
    ax_speed.set_xlim(0.1, 0.9) # Tighter bounds
    ax_speed.set_ylim(0.2, 0.9)

    # B. Map (Smaller than v1)
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=1, alpha=0.15)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=6)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=2, alpha=0.7)

    # C. Speed Graph (New!)
    ax_graph.set_title("SPEED PROFILE", color='white', fontsize=10, loc='left')
    # Plot the full background profile
    ax_graph.plot(df['time'], df['speed_kmh'], color='cyan', alpha=0.1, lw=1)
    # The active line and the "red ball"
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=0.9)
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=8)
    
    ax_graph.set_xlim(df['time'].min(), df['time'].max())
    ax_graph.set_ylim(0, df['speed_kmh'].max() * 1.1)
    ax_graph.tick_params(colors='white', labelsize=8)
    ax_graph.spines['bottom'].set_color('white')
    ax_graph.spines['left'].set_color('white')
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)

    def update(frame):
        if frame < len(df):
            row = df.iloc[frame]
            val = row['speed_kmh']
            curr_time = row['time']
            
            # 1. Update Speed
            speed_text.set_text(f"{int(val)}")
            progress = min(val / 60.0, 1.0)
            fill_idx = int(progress * 100)
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            
            # 2. Update Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail_start = max(0, frame - 36)
            map_path.set_data(df['lon'][trail_start:frame+1], df['lat'][trail_start:frame+1])
            
            # 3. Update Graph
            # We show the progress on the graph line
            graph_line.set_data(df['time'][:frame+1], df['speed_kmh'][:frame+1])
            graph_ball.set_data([curr_time], [val])
            
        return speed_text, speed_arc, map_tracker, map_path, graph_line, graph_ball

    num_frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    # Glow effects
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    mplcyberpunk.add_glow_effects(ax=ax_graph)
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud_v2.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=SAMPLE_FPS, bitrate=4000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error: {e}")
        save_path = save_path.replace(".mp4", ".gif")
        ani.save(save_path, writer='pillow', fps=SAMPLE_FPS)
        
    plt.close()
    return save_path

# --- RUN ---
print("Extracting telemetry for HUD v2...")
df_hud = parse_telemetry(INPUT_FILE)
print(f"Captured {len(df_hud)} telemetry rows.")

if not df_hud.empty:
    if not RENDER_FULL:
        print(f"Rendering HUD v2 mockup ({RENDER_SECONDS}s)...")
    render_racing_hud_v2(df_hud)
    print(f"\nSUCCESS! HUD v2 created in {OUTPUT_DIR}/racing_hud_v2.mp4")
else:
    print("Could not parse data.")
