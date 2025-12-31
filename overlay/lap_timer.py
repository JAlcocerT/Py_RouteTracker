import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- CONIFGURATION ---
# User mentioned switching to '1c' recently
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
VIDEO_DURATION_SEC = 532.5

# START LINE COORDINATES (Tweak these!)
# I will auto-populate these with the first point found in the file, 
# but you can overwrite them with specific coordinates if you want to move the line.
START_LAT = None 
START_LON = None
LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0 # Ignore crossings if they happen too soon (debounce)

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
                        'speed_kmh': speed * 3.6, # Assuming m/s input usually, but check if user wants km/h logic
                        # Actually previous scripts used SPEED_MULTIPLIER = 1.0 (indicating input is likely km/h or user handled it)
                        # Let's stick to 1.0 logic from v3b if data is already km/h?
                        # Wait, v3b: 'GPS Speed : 43.7' -> likely km/h if it's a go-kart? 
                        # Or m/s? 43 m/s = 150km/h. 43 km/h = 43 km/h. 
                        # Kart speed 43-85 km/h seems reasonable. 150 is too fast usually.
                        # BUT previous v3b used SPEED_MULTIPLIER = 1.0. I will check logic.
                        # Re-checking v3b: 'speed_kmh': speed * SPEED_MULTIPLIER
                        # If SPEED_MULTIPLIER was 1.0, then raw value is used.
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
            
            # Map to video duration (Blind Mapping)
            df['time'] = np.linspace(0, video_duration_sec, num_points)
            df = df.set_index('time').reset_index()
            return df
        except: continue
    return pd.DataFrame()

# --- 2. LAP TIMING LOGIC ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def detect_laps(df, start_lat, start_lon):
    laps = []
    
    # Calculate distance to Start Line for every point
    # Vectorized for speed? Or loop for state? Loop is easier for debounce logic.
    
    distances = []
    lap_start_time = df.iloc[0]['time']
    last_crossing_time = -MIN_LAP_TIME_SEC 
    
    lap_indices = [0] # List of indices where laps start
    
    print(f"Searching for crossings at {start_lat:.6f}, {start_lon:.6f} (Radius: {LAP_DETECTION_RADIUS_M}m)")
    
    # We loop through points
    # Logic: If we are INSIDE the radius AND enough time has passed since last lap
    # We count a lap.
    # To be precise, we want the point CLOSET to the center within that crossing window.
    # But simple threshold crossing is distinct enough for now.
    
    in_zone = False
    closest_dist_in_zone = 99999.0
    best_idx_in_zone = -1
    
    for i, row in df.iterrows():
        dist = haversine_distance(row['lat'], row['lon'], start_lat, start_lon)
        distances.append(dist)
        current_time = row['time']
        
        if dist < LAP_DETECTION_RADIUS_M:
            if not in_zone:
                # Entered zone
                if (current_time - last_crossing_time) > MIN_LAP_TIME_SEC:
                    in_zone = True
                    closest_dist_in_zone = dist
                    best_idx_in_zone = i
            else:
                # Inside zone, track closest point
                if dist < closest_dist_in_zone:
                    closest_dist_in_zone = dist
                    best_idx_in_zone = i
        else:
            if in_zone:
                # Exited zone. Register the BEST point as the crossing.
                in_zone = False
                lap_indices.append(best_idx_in_zone)
                last_crossing_time = df.iloc[best_idx_in_zone]['time']
                print(f"Lap detected at {last_crossing_time:.1f}s (Index {best_idx_in_zone})")

    # Calculate Lap Times
    lap_table = []
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
    df = parse_telemetry(FILE_PATH)
    if df.empty: exit(1)
    
    # Auto-pick defaults if None
    if START_LAT is None:
        START_LAT = df.iloc[0]['lat']
        START_LON = df.iloc[0]['lon']
        print(f"Auto-selected Start Line at start of file: {START_LAT}, {START_LON}")
        
    lap_stats, lap_indices, dists = detect_laps(df, START_LAT, START_LON)
    
    print("\n--- LAP TIMES ---")
    print(lap_stats)
    
    # --- PLOTTING ---
    try:
        import mplcyberpunk
        plt.style.use("cyberpunk")
    except:
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Plot Speed vs Time
    ax.plot(df['time'], df['raw_speed'], color='#00ff9f', lw=1.5, label='Speed')
    
    # 2. Vertical Lines for Laps
    for idx_val in lap_indices:
        t = df.iloc[idx_val]['time']
        ax.axvline(x=t, color='white', linestyle='--', alpha=0.5)
        # Label the line
        # ax.text(t, df['raw_speed'].max(), f"L{lap_indices.index(idx_val)+1}", 
        #         color='white', rotation=90, va='top', ha='right', fontsize=8)

    # Label Laps Centered
    for i, row in lap_stats.iterrows():
        mid_time = (row['Start Time'] + row['End Time']) / 2
        ax.text(mid_time, df['raw_speed'].max()*0.95, f"L{int(row['Lap'])}: {row['Duration']:.1f}s", 
                color='white', ha='center', fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.set_title(f"Lap Analysis: {len(lap_stats)} Laps Detected", color='white')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.2)
    
    plot_path = os.path.join(OUTPUT_DIR, "lap_analysis_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Analysis Plot: {plot_path}")
