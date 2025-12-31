import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import subprocess

# --- CONIFGURATION ---
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1c.txt"
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"

# START LINE CONFIGURATION
LAP_START_TIME_SEC = 5.0 

LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- AUTOMATIC DURATION ---
def get_video_duration(video_path):
    """Get the duration of a video file in seconds using ffprobe."""
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}. Using fallback.")
        return 532.5
        
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        print(f"Detected Video Duration: {duration:.2f}s")
        return duration
    except Exception as e:
        print(f"Error getting duration: {e}. Using fallback.")
        return 532.5 # Fallback

VIDEO_DURATION_SEC = get_video_duration(VIDEO_PATH)

# --- 1. DATA PARSING ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        dd = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        if parts[3].strip() in ['S', 'W']: dd *= -1
        return dd
    except: return None

def parse_telemetry(file_path, video_duration_sec=VIDEO_DURATION_SEC):
    encodings = ['utf-16le', 'utf-8', 'latin-1']
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
                    speed = float(m.group(1))
                    data.append({
                        'raw_speed': float(m.group(1)),
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
            
            num_points = len(df)
            if num_points < 2: return pd.DataFrame()
            
            # Map to video duration
            df['time'] = np.linspace(0, video_duration_sec, num_points)
            df = df.set_index('time').reset_index()
            return df
        except: continue
    return pd.DataFrame()

# --- 2. LAP TIMING LOGIC ---
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
    return row['lat'], row['lon'], row['time']

def detect_laps(df, start_lat, start_lon):
    laps = []
    distances = []
    lap_start_time = df.iloc[0]['time'] 
    
    lap_indices = []
    last_crossing_time = -MIN_LAP_TIME_SEC
    
    in_zone = False
    closest_dist_in_zone = 99999.0
    best_idx_in_zone = -1
    
    print(f"Detecting laps based on Start Line at {start_lat:.6f}, {start_lon:.6f}...")
    
    for i, row in df.iterrows():
        dist = haversine_distance(row['lat'], row['lon'], start_lat, start_lon)
        distances.append(dist)
        current_time = row['time']
        
        if (current_time - last_crossing_time) > MIN_LAP_TIME_SEC:
            if dist < LAP_DETECTION_RADIUS_M:
                if not in_zone:
                    in_zone = True
                    closest_dist_in_zone = dist
                    best_idx_in_zone = i
                else:
                    if dist < closest_dist_in_zone:
                        closest_dist_in_zone = dist
                        best_idx_in_zone = i
            else:
                if in_zone:
                    in_zone = False
                    lap_indices.append(best_idx_in_zone)
                    last_crossing_time = df.iloc[best_idx_in_zone]['time']
                    print(f"Crossing detected at {last_crossing_time:.1f}s")

    lap_table = []
    if not lap_indices: return pd.DataFrame(), [], []

    for k in range(1, len(lap_indices)):
        start_idx = lap_indices[k-1]
        end_idx = lap_indices[k]
        t_start = df.iloc[start_idx]['time']
        t_end = df.iloc[end_idx]['time']
        duration = t_end - t_start
        
        lap_slice = df.iloc[start_idx:end_idx]
        lap_table.append({
            'Lap': k,
            'Start Time': t_start,
            'End Time': t_end,
            'Duration': duration,
            'Avg Speed': lap_slice['raw_speed'].mean(),
            'Max Speed': lap_slice['raw_speed'].max(),
            'Min Speed': lap_slice['raw_speed'].min()
        })
        
    return pd.DataFrame(lap_table), lap_indices, distances

if __name__ == "__main__":
    print(f"Loading {FILE_PATH}...")
    df = parse_telemetry(FILE_PATH, VIDEO_DURATION_SEC)
    if df.empty: exit(1)
    
    s_lat, s_lon, actual_time = get_coordinates_at_time(df, LAP_START_TIME_SEC)
    print(f"\nTarget Start: {LAP_START_TIME_SEC}s (Lat: {s_lat:.6f}, Lon: {s_lon:.6f})")
    
    lap_stats, lap_indices, dists = detect_laps(df, s_lat, s_lon)
    
    print("\n--- LAP RESULTS ---")
    print(lap_stats)
    
    # --- PLOTTING ---
    try:
        import mplcyberpunk
        plt.style.use("cyberpunk")
    except:
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(14, 7)) 
    
    ax.plot(df['time'], df['raw_speed'], color='#00ff9f', lw=1.5, label='Speed')
    ax.axvline(x=LAP_START_TIME_SEC, color='yellow', linestyle=':', alpha=0.8, label='Defined Start')
    
    for idx_val in lap_indices:
        t = df.iloc[idx_val]['time']
        ax.axvline(x=t, color='white', linestyle='--', alpha=0.5)

    # LOOP THROUGH LAPS TO ANNOTATE MAX/MIN
    for k in range(1, len(lap_indices)):
        start_idx = lap_indices[k-1]
        end_idx = lap_indices[k]
        lap_slice = df.iloc[start_idx:end_idx]
        
        # Max Speed
        max_idx = lap_slice['raw_speed'].idxmax()
        max_val = lap_slice['raw_speed'].max()
        max_time = df.loc[max_idx, 'time']
        
        ax.annotate(f"{max_val:.2f}", 
                    xy=(max_time, max_val), 
                    xytext=(0, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#ff0055', lw=1.5),
                    color='#ff0055', ha='center', fontsize=8, fontweight='bold')

        # Min Speed 
        min_idx = lap_slice['raw_speed'].idxmin()
        min_val = lap_slice['raw_speed'].min()
        min_time = df.loc[min_idx, 'time']
        
        ax.annotate(f"{min_val:.2f}", 
                    xy=(min_time, min_val), 
                    xytext=(0, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5),
                    color='cyan', ha='center', fontsize=8, fontweight='bold')

        # Lap Label
        mid_time = (df.iloc[start_idx]['time'] + df.iloc[end_idx]['time']) / 2
        duration = df.iloc[end_idx]['time'] - df.iloc[start_idx]['time']
        ax.text(mid_time, df['raw_speed'].max()*1.05, f"L{k}\n{duration:.1f}s", 
                color='white', ha='center', fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    ax.set_title(f"Lap Analysis v4b (Auto-Duration) - Start @ {LAP_START_TIME_SEC}s", color='white')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    
    plot_path = os.path.join(OUTPUT_DIR, "lap_analysis_v4b.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Analysis Plot: {plot_path}")

    print("\n--- Youtube Video Chapters ---")
    print("00:00 - Intro")
    for i, row in lap_stats.iterrows():
        seconds = int(row['Start Time'])
        m, s = divmod(seconds, 60)
        print(f"{m:02d}:{s:02d} - Lap {int(row['Lap'])} ({row['Duration']:.2f}s)")
