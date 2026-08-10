import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import subprocess

# --- CONFIGURATION ---
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020411.MP4" #generates the telemetry txt with the GPS data automatically via exiftool
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"

# START LINE CONFIGURATION
LAP_START_TIME_SEC = 40.0 
LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

# FEATURES
SLICE_BEST_LAP = True
SLICE_BUFFER_SEC = 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. AUTO EXIF EXTRACTION ---
def extract_telemetry(video_path):
    """
    Runs exiftool -ee on the video to generate a .txt file.
    Returns the path to the generated txt file.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Telemetry file will be saved in the same folder as the video or OUTPUT_DIR?
    # User mentioned "generate its related FILE_PATH".
    # Let's save it in OUTPUT_DIR to keep things organized or beside the video.
    # User's previous files were in Z_GoPro. Let's put it in Z_GoPro (input dir).
    video_dir = os.path.dirname(video_path)
    txt_path = os.path.join(video_dir, f"{base_name}_telemetry.txt")
    
    if os.path.exists(txt_path):
        print(f"Telemetry file found: {txt_path}")
        return txt_path
        
    print(f"Extracting telemetry from {video_path}...")
    print(f"output -> {txt_path}")
    
    try:
        # exiftool -ee "VIDEO_PATH" > "TXT_PATH"
        with open(txt_path, "w") as outfile:
            subprocess.run(["exiftool", "-ee", video_path], stdout=outfile, check=True)
        print("Extraction complete.")
        return txt_path
    except FileNotFoundError:
        print("Error: 'exiftool' not found. Please install it (sudo apt install libimage-exiftool-perl).")
        exit(1)
    except Exception as e:
        print(f"Extraction failed: {e}")
        exit(1)

# --- 2. AUTO DURATION ---
def get_video_duration(video_path):
    if not os.path.exists(video_path):
        return 532.5 # Fallback
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return 532.5

# --- 3. DATA PARSING ---
def dms_to_dd(dms_str):
    if not dms_str: return None
    try:
        parts = re.split(r'[deg\'"]+', dms_str)
        dd = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        if parts[3].strip() in ['S', 'W']: dd *= -1
        return dd
    except: return None

def parse_telemetry(file_path, video_duration_sec):
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
            
            df['time'] = np.linspace(0, video_duration_sec, num_points)
            df = df.set_index('time').reset_index()
            return df
        except: continue
    return pd.DataFrame()

# --- 4. ENGINE ---
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
    lap_indices = []
    last_crossing_time = -MIN_LAP_TIME_SEC
    in_zone = False
    closest_dist_in_zone = 99999.0
    best_idx_in_zone = -1
    
    print(f"Detecting laps based on Start Line at {start_lat:.6f}, {start_lon:.6f}...")
    
    for i, row in df.iterrows():
        dist = haversine_distance(row['lat'], row['lon'], start_lat, start_lon)
        if (row['time'] - last_crossing_time) > MIN_LAP_TIME_SEC:
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
        lap_slice = df.iloc[start_idx:end_idx]
        
        lap_table.append({
            'Lap': k,
            'Start Time': df.iloc[start_idx]['time'],
            'End Time': df.iloc[end_idx]['time'],
            'Duration': df.iloc[end_idx]['time'] - df.iloc[start_idx]['time'],
            'Avg Speed': lap_slice['raw_speed'].mean(),
            'Max Speed': lap_slice['raw_speed'].max(),
            'Min Speed': lap_slice['raw_speed'].min()
        })
        
    return pd.DataFrame(lap_table), lap_indices, []

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        exit(1)

    # 1. Get Telemetry File
    txt_path = extract_telemetry(VIDEO_PATH)
    
    # 2. Get Duration
    duration = get_video_duration(VIDEO_PATH)
    print(f"Detected Duration: {duration:.2f}s")
    
    # 3. Parse
    df = parse_telemetry(txt_path, duration)
    if df.empty: exit(1)
    
    # 4. Detect Laps
    s_lat, s_lon, actual_time = get_coordinates_at_time(df, LAP_START_TIME_SEC)
    print(f"\nTarget Start: {LAP_START_TIME_SEC}s (Lat: {s_lat:.6f}, Lon: {s_lon:.6f})")
    
    lap_stats, lap_indices, _ = detect_laps(df, s_lat, s_lon)
    
    print("\n--- LAP RESULTS ---")
    print(lap_stats)
    
    # 5. Plot
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

    for k in range(1, len(lap_indices)):
        start_idx = lap_indices[k-1]; end_idx = lap_indices[k]
        lap_slice = df.iloc[start_idx:end_idx]
        
        # Max
        max_idx = lap_slice['raw_speed'].idxmax(); max_val = lap_slice['raw_speed'].max()
        ax.annotate(f"{max_val:.2f}", xy=(df.loc[max_idx, 'time'], max_val), xytext=(0, 15), 
                    textcoords='offset points', arrowprops=dict(arrowstyle='->', color='#ff0055', lw=1.5), 
                    color='#ff0055', ha='center', fontsize=8, fontweight='bold')
        # Min
        min_idx = lap_slice['raw_speed'].idxmin(); min_val = lap_slice['raw_speed'].min()
        ax.annotate(f"{min_val:.2f}", xy=(df.loc[min_idx, 'time'], min_val), xytext=(0, -20), 
                    textcoords='offset points', arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5), 
                    color='cyan', ha='center', fontsize=8, fontweight='bold')
        # Label
        mid_time = (df.iloc[start_idx]['time'] + df.iloc[end_idx]['time']) / 2
        ax.text(mid_time, df['raw_speed'].max()*1.05, f"L{k}\n{lap_stats.iloc[k-1]['Duration']:.1f}s", 
                color='white', ha='center', fontweight='bold', fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    plot_path = os.path.join(OUTPUT_DIR, "lap_analysis_v5.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Analysis Plot: {plot_path}")

    # 6. Best Lap Slice
    if SLICE_BEST_LAP and not lap_stats.empty:
        best_lap_row = lap_stats.loc[lap_stats['Duration'].idxmin()]
        best_lap_num = int(best_lap_row['Lap'])
        t_start = max(0, best_lap_row['Start Time'] - SLICE_BUFFER_SEC)
        t_end = min(duration, best_lap_row['End Time'] + SLICE_BUFFER_SEC)
        
        output_filename = f"Best_Lap_{best_lap_num}_{best_lap_row['Duration']:.2f}s_v5.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"\n--- BEST LAP ({best_lap_num}): Slicing {t_start:.2f}s -> {t_end:.2f}s ---")
        cmd = ["ffmpeg", "-y", "-ss", f"{t_start}", "-i", VIDEO_PATH, "-t", f"{t_end-t_start}", "-c", "copy", output_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"SUCCESS: {output_path}")
        except: print("Slice Failed.")

    # 7. Chapters
    print("\n--- Youtube Video Chapters ---")
    print("00:00 - Intro")
    for i, row in lap_stats.iterrows():
        seconds = int(row['Start Time']); m, s = divmod(seconds, 60)
        marker = "🔥 BEST LAP" if (SLICE_BEST_LAP and int(row['Lap']) == lap_stats['Duration'].idxmin()+1) else ""
        print(f"{m:02d}:{s:02d} - Lap {int(row['Lap'])} ({row['Duration']:.2f}s) {marker}")
