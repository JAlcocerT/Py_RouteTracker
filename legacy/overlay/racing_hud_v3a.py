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
FILE_REF = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt" 
FILE_CURR = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-2b.txt"
LABEL_REF = "Run 1"
LABEL_CURR = "Run 2"

OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
SAMPLE_FPS = 18
RENDER_SECONDS = 25 #for testing
RENDER_FULL = True #False for testing 
SPEED_MULTIPLIER = 1.0 
MAX_EXPECTED_SPEED = 85 

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
                    df['lat'] = df['lat'].ffill()
                    df['lon'] = df['lon'].ffill()
                    df = df.dropna(subset=['speed_kmh', 'lat', 'lon'])
                    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
                    return df
        except: continue
    return pd.DataFrame()

# --- 3. COMPARATIVE HUD RENDER ---
def render_racing_hud_v3a(df_ref, df_curr):
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 5), dpi=100) # Slightly taller for the table
    gs = GridSpec(1, 4, figure=fig)
    
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[0, 2:])
    
    fig.patch.set_alpha(0.0)
    for ax in [ax_speed, ax_map, ax_graph]: ax.patch.set_alpha(0.0)
    ax_speed.axis('off'); ax_map.axis('off')

    # --- ADDING BACKGROUND PANELS FOR READABILITY ---
    # Panel 1: Behind Speedometer & Map
    # (x, y) = (-0.1, -0.1) relative to ax_speed, width=2.4 (covers map too), height=1.2
    bg_panel_1 = patches.Rectangle((-0.1, -0.1), 2.5, 1.2, transform=ax_speed.transAxes, 
                                   color='black', alpha=0.6, zorder=-10)
    ax_speed.add_patch(bg_panel_1)

    # Panel 2: Behind Graph
    # (x, y) = (-0.05, -0.5) to cover table, width=1.1, height=1.6
    bg_panel_2 = patches.Rectangle((-0.05, -0.55), 1.1, 1.65, transform=ax_graph.transAxes, 
                                   color='black', alpha=0.6, zorder=-10)
    ax_graph.add_patch(bg_panel_2)

    # A. ULTRA SPEEDOMETER (Compact)
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

    # B. Compact Map
    ax_map.plot(df_curr['lon'], df_curr['lat'], color='cyan', lw=1, alpha=0.1)
    map_tracker, = ax_map.plot([], [], 'o', color='#ff0055', markersize=5)
    map_path, = ax_map.plot([], [], color='#00ff9f', lw=2, alpha=0.8)

    # C. SPEED COMPARISON GRAPH
    ax_graph.set_title(f"SPEED COMPARISON: {LABEL_REF} vs {LABEL_CURR}", color='white', fontsize=10, loc='center', pad=10)
    
    # 1. Plot Reference (Ghost)
    ax_graph.plot(df_ref['time'], df_ref['speed_kmh'], color='white', alpha=0.15, lw=1, label=LABEL_REF)
    ref_avg = df_ref['speed_kmh'].mean()
    ax_graph.axhline(y=ref_avg, color='white', linestyle='--', alpha=0.2, lw=0.5)
    
    # 2. Plot Current (Live)
    graph_line, = ax_graph.plot([], [], color='#00ff9f', lw=2, alpha=1.0, label=LABEL_CURR)
    curr_avg_line = ax_graph.axhline(y=0, color='#00ff9f', linestyle='--', alpha=0.3, lw=0.5) # Will update
    
    # Tracker ball
    graph_ball, = ax_graph.plot([], [], 'o', color='#ff0055', markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # Axis & Limits
    max_time = max(df_ref['time'].max(), df_curr['time'].max())
    max_speed = max(df_ref['speed_kmh'].max(), df_curr['speed_kmh'].max())
    ax_graph.set_xlim(0, max_time)
    ax_graph.set_ylim(0, max_speed * 1.1)
    ax_graph.tick_params(colors='white', labelsize=8)
    for spine in ax_graph.spines.values(): 
        spine.set_edgecolor('white')
        spine.set_visible(True) # Force visibility
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    ax_graph.grid(True, alpha=0.1, color='white', linestyle=':')

    # 3. Stats Table
    # We'll update the 'Current' stats live, 'Ref' stats are static
    ref_stats = [f"{df_ref['speed_kmh'].max():.1f}", f"{ref_avg:.1f}"]
    
    table_data = [
        ['Max Speed', ref_stats[0], '-'],
        ['Avg Speed', ref_stats[1], '-']
    ]
    table = ax_graph.table(cellText=table_data, colLabels=['Stat', LABEL_REF, LABEL_CURR],
                           loc='bottom', bbox=[0.1, -0.45, 0.8, 0.3], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        cell.set_text_props(color='white')
        cell.set_edgecolor('white')
        cell.set_facecolor('none')
    
    # Adjust layout to make room for table
    plt.subplots_adjust(bottom=0.3)

    peak_speed = 0.0

    def update(frame):
        nonlocal peak_speed
        if frame < len(df_curr):
            row = df_curr.iloc[frame]
            val = row['speed_kmh']
            peak_speed = max(peak_speed, val)
            
            # Update Speedometer
            speed_text.set_text(f"{int(val)}")
            ratio = min(val / MAX_EXPECTED_SPEED, 1.0)
            fill_idx = int(ratio * 100)
            color = '#00ff9f' if ratio < 0.5 else '#ffff00' if ratio < 0.8 else '#ff0055'
            speed_arc.set_data(arc_x[:fill_idx], arc_y[:fill_idx])
            speed_arc.set_color(color)
            
            peak_ratio = min(peak_speed / MAX_EXPECTED_SPEED, 1.0)
            peak_rad = np.deg2rad(180 - peak_ratio * 180)
            peak_marker.set_data([0.5 + (arc_radius+0.05)*np.cos(peak_rad)], [0.5 + (arc_radius+0.05)*np.sin(peak_rad)])

            # Update Map
            map_tracker.set_data([row['lon']], [row['lat']])
            trail = max(0, frame - 30)
            map_path.set_data(df_curr['lon'][trail:frame+1], df_curr['lat'][trail:frame+1])
            
            # Update Graph
            current_data = df_curr.iloc[:frame+1]
            graph_line.set_data(current_data['time'], current_data['speed_kmh'])
            graph_ball.set_data([row['time']], [val])
            
            # Update Avg Line
            curr_avg = current_data['speed_kmh'].mean()
            curr_avg_line.set_ydata([curr_avg, curr_avg])
            
            # Update Table
            table.get_celld()[(1, 2)].get_text().set_text(f"{peak_speed:.1f}")
            table.get_celld()[(2, 2)].get_text().set_text(f"{curr_avg:.1f}")

        return speed_text, speed_arc, peak_marker, map_tracker, map_path, graph_line, graph_ball, curr_avg_line

    num_frames = len(df_curr) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False) # blit=False for table text updates
    
    mplcyberpunk.add_glow_effects(ax=ax_speed)
    mplcyberpunk.add_glow_effects(ax=ax_map)
    # mplcyberpunk.add_glow_effects(ax=ax_graph) # Skipped on graph to keep reference line clear
    
    save_path = os.path.join(OUTPUT_DIR, "racing_hud_v3a.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=SAMPLE_FPS, bitrate=5000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error: {e}")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=SAMPLE_FPS)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("Extracting Reference Data...")
    df1 = parse_telemetry(FILE_REF)
    print(f"Loaded {len(df1)} rows for {LABEL_REF}")
    
    print("Extracting Current Data...")
    df2 = parse_telemetry(FILE_CURR)
    print(f"Loaded {len(df2)} rows for {LABEL_CURR}")
    
    if not df1.empty and not df2.empty:
        render_racing_hud_v3a(df1, df2)
        print(f"\nSUCCESS! Comparative HUD created in {OUTPUT_DIR}/racing_hud_v3a.mp4")
    else:
        print("Error: Could not load one or both data files.")
