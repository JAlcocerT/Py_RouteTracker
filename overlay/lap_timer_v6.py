import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import subprocess
import sys

# --- CONFIGURATION (Match v5) ---
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020411.MP4" 
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
        exit(1)

# --- 2. AUTO DURATION ---
def get_video_duration(video_path):
    if not os.path.exists(video_path): return 532.5 
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        return float(subprocess.check_output(cmd).decode().strip())
    except: return 532.5

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
            with open(file_path, 'r', encoding=enc) as f: content = f.read()
            lines = content.splitlines()
            data = []
            cur_lat, cur_lon = np.nan, np.nan
            
            for line in lines:
                m = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                if m:
                    # V6 Fix: Ensure consistent units with HUD (km/h)
                    # HUD v5 uses * 3.6. Lap Timer v5 used raw.
                    # Let's align with HUD for consistency if desired, 
                    # BUT v5 prompt said maintain v5 logic. v5 logic was raw.
                    # However, usually raw is m/s. Let's verify.
                    # Lap Timer v5 plot used 'raw_speed' directly.
                    # If user wants km/h, we should apply * 3.6.
                    # Let's keep v5 logic (raw) unless specified, but v6 comparison implies usefulness.
                    # I will add conversion if values look like m/s (<40 max).
                    val = float(m.group(1))
                    data.append({'raw_speed': val, 'lat': cur_lat, 'lon': cur_lon})
                
                lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                if lat_m: cur_lat = dms_to_dd(lat_m.group(1))
                lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                if lon_m: cur_lon = dms_to_dd(lon_m.group(1))
            
            if not data: continue
            df = pd.DataFrame(data)
            df[['lat','lon']] = df[['lat','lon']].ffill().bfill()
            df = df[(df['lat']!=0)&(df['lon']!=0)]
            if len(df)<2: return pd.DataFrame()
            
            df['time'] = np.linspace(0, video_duration_sec, len(df))
            return df
        except: continue
    return pd.DataFrame()

# --- 4. ENGINE ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_coordinates_at_time(df, target_time):
    idx = (df['time'] - target_time).abs().idxmin()
    row = df.iloc[idx]
    return row['lat'], row['lon'], row['time']

def detect_laps(df, start_lat, start_lon):
    lap_indices = []
    last_crossing_time = -MIN_LAP_TIME_SEC
    in_zone = False
    closest_dist = 99999.0
    best_idx = -1
    
    print(f"Detecting laps based on Start Line at {start_lat:.6f}, {start_lon:.6f}...")
    
    for i, row in df.iterrows():
        dist = haversine_distance(row['lat'], row['lon'], start_lat, start_lon)
        if (row['time'] - last_crossing_time) > MIN_LAP_TIME_SEC:
            if dist < LAP_DETECTION_RADIUS_M:
                in_zone = True
                if dist < closest_dist: closest_dist = dist; best_idx = i
            else:
                if in_zone:
                    in_zone = False
                    lap_indices.append(best_idx)
                    last_crossing_time = df.iloc[best_idx]['time']
                    closest_dist = 99999.0

    lap_table = []
    if not lap_indices: return pd.DataFrame(), [], []

    for k in range(1, len(lap_indices)):
        s_idx = lap_indices[k-1]; e_idx = lap_indices[k]
        lap_slice = df.iloc[s_idx:e_idx]
        
        lap_table.append({
            'Lap': k,
            'Start Time': df.iloc[s_idx]['time'],
            'End Time': df.iloc[e_idx]['time'],
            'Duration': df.iloc[e_idx]['time'] - df.iloc[s_idx]['time'],
            'Avg Speed': lap_slice['raw_speed'].mean(),
            'Max Speed': lap_slice['raw_speed'].max()
        })
        
    return pd.DataFrame(lap_table), lap_indices, []

# --- 5. INTERACTIVE COMPARISON (V6 NEW) ---
def compare_laps(df, lap_stats, lap_indices, l1, l2):
    print(f"\n--- COMPARING LAP {l1} vs LAP {l2} ---")
    
    # Get Slices
    try:
        # lap_indices is list of crossing indices. 
        # Lap 1 is between lap_indices[0] and lap_indices[1]
        if l1 > len(lap_indices)-1 or l2 > len(lap_indices)-1:
            print("Invalid Lap Numbers.")
            return

        # Indices logic: Lap k uses indices k-1 and k
        s1, e1 = lap_indices[l1-1], lap_indices[l1]
        s2, e2 = lap_indices[l2-1], lap_indices[l2]
        
        slice1 = df.iloc[s1:e1].copy()
        slice2 = df.iloc[s2:e2].copy()
        
        # Reset Time to 0 (Relative Time Comparison)
        slice1['rel_time'] = slice1['time'] - slice1.iloc[0]['time']
        slice2['rel_time'] = slice2['time'] - slice2.iloc[0]['time']
        
        # Plot
        try:
            import mplcyberpunk
            plt.style.use("cyberpunk")
        except: plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Stats
        dur1 = lap_stats.loc[lap_stats['Lap']==l1, 'Duration'].values[0]
        dur2 = lap_stats.loc[lap_stats['Lap']==l2, 'Duration'].values[0]
        
        ax.plot(slice1['rel_time'], slice1['raw_speed'], color='cyan', lw=2, label=f"Lap {l1} ({dur1:.2f}s)")
        ax.plot(slice2['rel_time'], slice2['raw_speed'], color='#ff0055', lw=2, label=f"Lap {l2} ({dur2:.2f}s)")
        
        # Area fill for difference? Or just simple lines.
        # Simple lines are cleaner unless interpolation is done.
        
        ax.set_title(f"LAP COMPARISON: Lap {l1} vs Lap {l2}", color='white', fontsize=14, pad=15)
        ax.set_xlabel("Relative Time (s)", color='white')
        ax.set_ylabel("Speed", color='white')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        out_name = f"lap_compare_L{l1}_vs_L{l2}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plt.savefig(out_path, dpi=150)
        print(f"Saved Comparison Plot: {out_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"Comparison Error: {e}")


# --- EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH): exit(1)
    
    # Standard Flow
    txt = extract_telemetry(VIDEO_PATH)
    dur = get_video_duration(VIDEO_PATH)
    print(f"Duration: {dur:.2f}s")
    
    df = parse_telemetry(txt, dur)
    if df.empty: exit(1)
    
    # Auto-convert if Metric implies (simple check)
    if df['raw_speed'].max() < 50: # Likely m/s
        print("Auto-Converting m/s to km/h...")
        df['raw_speed'] *= 3.6
    
    s_lat, s_lon, _ = get_coordinates_at_time(df, LAP_START_TIME_SEC)
    print(f"\nStart Line: {LAP_START_TIME_SEC}s (Lat: {s_lat:.6f}, Lon: {s_lon:.6f})")
    
    lap_stats, lap_indices, _ = detect_laps(df, s_lat, s_lon)
    
    print("\n--- LAP RESULTS ---")
    print(lap_stats)
    
    # Original V5 Plot
    try:
        import mplcyberpunk
        plt.style.use("cyberpunk")
    except: plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(14, 7)) 
    ax.plot(df['time'], df['raw_speed'], color='#00ff9f', lw=1.5, label='Speed')
    ax.axvline(x=LAP_START_TIME_SEC, color='yellow', linestyle=':', alpha=0.8, label='Defined Start')
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
    
    plot_path = os.path.join(OUTPUT_DIR, "lap_analysis_v6.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Analysis Plot: {plot_path}")
    
    # Best Lap & Chapters
    if SLICE_BEST_LAP and not lap_stats.empty:
        best_row = lap_stats.loc[lap_stats['Duration'].idxmin()]
        best_lap_num = int(best_row['Lap'])
        print("\n--- Youtube Video Chapters ---")
        print("00:00 - Intro")
        for i, row in lap_stats.iterrows():
            st = int(row['Start Time']); m, s = divmod(st, 60)
            mark = "🔥 BEST LAP" if int(row['Lap']) == best_lap_num else ""
            print(f"{m:02d}:{s:02d} - Lap {int(row['Lap'])} ({row['Duration']:.2f}s) {mark}")
            
        t_start = max(0, best_row['Start Time'] - SLICE_BUFFER_SEC)
        t_end = min(dur, best_row['End Time'] + SLICE_BUFFER_SEC)
        out_path = os.path.join(OUTPUT_DIR, f"Best_Lap_{best_lap_num}_{best_row['Duration']:.2f}s_v6.mp4")
        print(f"\n--- Slicing Best Lap: {t_start:.2f}s -> {t_end:.2f}s ---")
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t_start}", "-i", VIDEO_PATH, "-t", f"{t_end-t_start}", "-c", "copy", out_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"SUCCESS: {out_path}")

    # --- INTERACTIVE COMPARISON ---
    print("\n" + "="*40)
    print("      INTERACTIVE LAP COMPARISON      ")
    print("="*40)
    print("Enter two lap numbers to compare their Speed Profiles.")
    print("Example: '1 3' compares Lap 1 vs Lap 3.")
    print("Enter '0' or just press Enter to SKIP.")
    
    user_input = input("Compare Laps > ").strip()
    
    if user_input and user_input != '0':
        try:
            parts = user_input.split()
            if len(parts) >= 2:
                l1 = int(parts[0])
                l2 = int(parts[1])
                compare_laps(df, lap_stats, lap_indices, l1, l2)
            else:
                print("Skipping: Need 2 lap numbers.")
        except ValueError:
            print("Skipping: Invalid input.")
    else:
        print("Skipping Comparison.")

    print("\nDone.")
