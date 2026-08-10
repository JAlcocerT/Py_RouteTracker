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
RENDER_SECONDS = 10 # Default for testing
RENDER_FULL = True #False 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. DATA PARSING ---
def dms_to_dd(dms_str):
    """Convert '37 deg 33\' 32.17" N' to decimal degrees."""
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
                        # Extract Time
                        m = re.search(r"Sample Time\s+:\s+([\d\.]+)\s+s", line)
                        if m: current_sample_time = float(m.group(1))
                        
                        # Extract Speed
                        m = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                        if m:
                            speed = float(m.group(1))
                            micro_offset = len([d for d in data if d['sample_time'] == current_sample_time]) * (1.0/SAMPLE_FPS)
                            data.append({
                                'time': current_sample_time + micro_offset,
                                'sample_time': current_sample_time,
                                'speed_kmh': speed * 1.0, # km/h
                                'lat': None,
                                'lon': None
                            })
                            
                        # Extract Lat/Lon (attach to latest data point)
                        if data:
                            lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                            if lat_m: data[-1]['lat'] = dms_to_dd(lat_m.group(1))
                            
                            lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                            if lon_m: data[-1]['lon'] = dms_to_dd(lon_m.group(1))
                            
                    df = pd.DataFrame(data)
                    # Forward fill coordinates since they might not be on the same row in the txt
                    df['lat'] = df['lat'].ffill()
                    df['lon'] = df['lon'].ffill()
                    return df.dropna(subset=['speed_kmh'])
        except Exception:
            continue
    return pd.DataFrame()

# --- 3. PREMIUM HUD RENDER ---
def render_racing_hud(df):
    plt.style.use('cyberpunk')
    # Create two subplots: Speed (Left) and Map (Right)
    fig, (ax_speed, ax_map) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    fig.patch.set_alpha(0.0)
    
    for ax in [ax_speed, ax_map]:
        ax.patch.set_alpha(0.0)
        ax.axis('off')

    # A. Speedometer Design (Arc)
    theta = np.linspace(np.pi, 0, 100) # Semi-circle arc
    arc_x = 0.5 + 0.4 * np.cos(theta)
    arc_y = 0.5 + 0.4 * np.sin(theta)
    
    ax_speed.plot(arc_x, arc_y, color='white', lw=1, alpha=0.3) # Static background arc
    speed_arc, = ax_speed.plot([], [], color='#00ff9f', lw=8, alpha=0.9)
    speed_text = ax_speed.text(0.5, 0.45, '0', fontsize=60, color='white', ha='center', fontweight='bold')
    ax_speed.text(0.5, 0.3, 'KM/H', fontsize=14, color='#00ff9f', ha='center')

    # B. Map Design
    # Pre-plot the full path in dim blue
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=2, alpha=0.1)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=8)
    # Highlight the current lap/segment (last 2 seconds)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=3, alpha=0.8)

    def update(frame):
        if frame < len(df):
            row = df.iloc[frame]
            val = row['speed_kmh']
            
            # Update Speedometer
            speed_text.set_text(f"{int(val)}")
            progress = min(val / 60.0, 1.0) # Assume 60 max kmh for the arc
            fill_idx = int(progress * 100)
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            
            # Update Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail_start = max(0, frame - 36)
            map_path.set_data(df['lon'][trail_start:frame+1], df['lat'][trail_start:frame+1])
            
        return speed_text, speed_arc, map_tracker, map_path

    num_frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=SAMPLE_FPS, bitrate=3000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error: {e}")
        save_path = save_path.replace(".mp4", ".gif")
        ani.save(save_path, writer='pillow', fps=SAMPLE_FPS)
        
    plt.close()
    return save_path

# --- RUN ---
print("Extracting full telemetry (Speed + GPS)...")
df_hud = parse_telemetry(INPUT_FILE)
print(f"Captured {len(df_hud)} telemetry rows.")

if not df_hud.empty:
    if not RENDER_FULL:
        print(f"Rendering basic mockup ({RENDER_SECONDS}s)...")
    render_racing_hud(df_hud)
    print(f"\nSUCCESS! Premium HUD created in {OUTPUT_DIR}/racing_hud.mp4")
else:
    print("Could not parse data. Check file content.")
