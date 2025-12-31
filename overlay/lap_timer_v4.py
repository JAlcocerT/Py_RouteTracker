import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- CONIFGURATION ---
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
VIDEO_DURATION_SEC = 532.5

# START LINE CONFIGURATION
# Instead of strict coords, we define WHERE the lap starts by TIME.
# e.g. "I crossed the line at 00:13 in the video"
LAP_START_TIME_SEC = 13.0 

LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """Finds the lat/lon closest to the given timestamp."""
    # Find index where time is closest to target_time
    idx = (df['time'] - target_time).abs().idxmin()
    row = df.iloc[idx]
    return row['lat'], row['lon'], row['time']

def detect_laps(df, start_lat, start_lon):
    laps = []
    distances = []
    lap_start_time = df.iloc[0]['time'] # Default catch
    
    # We start counting laps AFTER the first crossing if we define Start Line mid-lap?
    # Actually, standard logic: 
    # 1. We start the timer when we cross the start line the FIRST time?
    #    OR does the video start AT the "Lap Start"?
    # The user set LAP_START_TIME_SEC = 13.0.
    # This implies the first crossing is AT 13.0.
    # So Lap 1 effectively starts at 13.0.
    
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
        
        # Debounce: Only look for crossings if enough time passed since last one
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
                    # Exited zone -> Register Lap
                    in_zone = False
                    lap_indices.append(best_idx_in_zone)
                    last_crossing_time = df.iloc[best_idx_in_zone]['time']
                    print(f"Crossing detected at {last_crossing_time:.1f}s")

    # Build Lap Table
    lap_table = []
    # If the user defines the start line at t=13.0, that is effectively the START of Lap 1?
    # Or is Lap 1 the "Out Lap"?
    # The first detected crossing will be very close to LAP_START_TIME_SEC (e.g. 13.0s).
    # Subsequent crossings define closed laps.
    
    if not lap_indices:
        return pd.DataFrame(), [], []

    for k in range(1, len(lap_indices)):
        start_idx = lap_indices[k-1]
        end_idx = lap_indices[k]
        t_start = df.iloc[start_idx]['time']
        t_end = df.iloc[end_idx]['time']
        duration = t_end - t_start
        
        lap_table.append({
            'Lap': k,
            'Start Time': t_start,
            'End Time': t_end,
            'Duration': duration,
            'Avg Speed': df.iloc[start_idx:end_idx]['raw_speed'].mean()
        })
        
    return pd.DataFrame(lap_table), lap_indices, distances

if __name__ == "__main__":
    print(f"Loading {FILE_PATH}...")
    df = parse_telemetry(FILE_PATH)
    if df.empty: 
        print("Error: No data.")
        exit(1)
    
    # 1. Determine Start Coordinates from Time
    s_lat, s_lon, actual_time = get_coordinates_at_time(df, LAP_START_TIME_SEC)
    print(f"\n--- START LINE SETUP ---")
    print(f"Target Time: {LAP_START_TIME_SEC}s")
    print(f"Found GPS Coordinate at {actual_time:.2f}s:")
    print(f"Lat: {s_lat:.6f}, Lon: {s_lon:.6f}")
    
    # 2. Detect Laps using this anchor
    lap_stats, lap_indices, dists = detect_laps(df, s_lat, s_lon)
    
    print("\n--- LAP RESULTS ---")
    print(lap_stats)
    
    # 3. Plot
    try:
        import mplcyberpunk
        plt.style.use("cyberpunk")
    except:
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['time'], df['raw_speed'], color='#00ff9f', lw=1.5, label='Speed')
    
    # Mark user-defined Start Time
    ax.axvline(x=LAP_START_TIME_SEC, color='yellow', linestyle=':', alpha=0.8, label='Defined Start')
    
    # Mark Detected Crossings
    for idx_val in lap_indices:
        t = df.iloc[idx_val]['time']
        ax.axvline(x=t, color='white', linestyle='--', alpha=0.5)

    # Label Laps
    for i, row in lap_stats.iterrows():
        mid_time = (row['Start Time'] + row['End Time']) / 2
        ax.text(mid_time, df['raw_speed'].max()*0.95, f"L{int(row['Lap'])}: {row['Duration']:.1f}s", 
                color='white', ha='center', fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    ax.set_title(f"Lap Analysis (Start @ {LAP_START_TIME_SEC}s) - {len(lap_stats)} Laps", color='white')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    
    plot_path = os.path.join(OUTPUT_DIR, "lap_analysis_v4.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Analysis Plot: {plot_path}")
