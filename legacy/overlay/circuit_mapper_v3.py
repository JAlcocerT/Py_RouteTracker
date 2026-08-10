import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- CONFIGURATION ---
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-2b.txt" 
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
TRACK_WIDTH_METERS = 8.0 
WIDTH_HALF = TRACK_WIDTH_METERS / 2.0

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

def parse_lat_lon(file_path):
    print(f"Reading {file_path}...")
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    data = []
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            lines = content.splitlines()
            cur_lat, cur_lon = np.nan, np.nan
            
            for line in lines:
                lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                if lat_m: cur_lat = dms_to_dd(lat_m.group(1))
                lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                if lon_m: cur_lon = dms_to_dd(lon_m.group(1))
                
                if not np.isnan(cur_lat) and not np.isnan(cur_lon):
                    if not data or (data[-1][0] != cur_lat or data[-1][1] != cur_lon):
                        data.append([cur_lat, cur_lon])
            
            if data:
                return pd.DataFrame(data, columns=['lat', 'lon'])
        except: continue
    return pd.DataFrame()

# --- 2. GEOMETRY & LAP EXTRACTION ---
def latlon_to_meters(df):
    lat_mean = df['lat'].mean()
    lat_scale = 111132.954
    lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mean))
    
    y = (df['lat'] - lat_mean) * lat_scale
    x = (df['lon'] - df['lon'].mean()) * lon_scale
    return x.to_numpy(), y.to_numpy()

def extract_single_lap(x, y):
    """
    Attempts to find the first complete lap closure.
    1. Defines Start Point (x[0], y[0])
    2. Searches for when we return to within 'threshold' meters of Start
    3. Ignores the first 'min_points' to avoid immediate triggers
    """
    start_x, start_y = x[0], y[0]
    min_dist_threshold = 10.0 # meters (Capture closure within 10m)
    min_points_buffer = 500   # Ignore first N points to let the kart leave the start
    
    if len(x) < min_points_buffer:
        print("Not enough points to find a lap.")
        return x, y
        
    print("Attempting to detect automatic lap closure...")
    
    for i in range(min_points_buffer, len(x)):
        dist = np.sqrt((x[i] - start_x)**2 + (y[i] - start_y)**2)
        if dist < min_dist_threshold:
            print(f"Lap Closure Detected at Index {i}! (Distance to start: {dist:.2f}m)")
            # Return the slice + 1 to close the loop
            return x[:i+1], y[:i+1]
            
    print("Warning: No loop closure detected. Using full track (Open Sprint?).")
    return x, y

def calculate_canonical_boundaries(x, y, width):
    # Smooth a bit for clean geometry
    window = 10
    if len(x) > window:
        x_smooth = np.convolve(x, np.ones(window)/window, mode='same')
        y_smooth = np.convolve(y, np.ones(window)/window, mode='same')
    else:
        x_smooth, y_smooth = x, y

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    lengths = np.sqrt(dx**2 + dy**2)
    lengths[lengths == 0] = 1.0
    
    ux = dx / lengths
    uy = dy / lengths
    
    # Normals
    nx = -uy
    ny = ux
    
    half = width / 2.0
    lx = x + nx * half
    ly = y + ny * half
    rx = x - nx * half
    ry = y - ny * half
    
    return (lx, ly), (rx, ry)

if __name__ == "__main__":
    df = parse_lat_lon(FILE_PATH)
    if df.empty: exit(1)
    
    # 1. Get raw meter coordinates
    mx_all, my_all = latlon_to_meters(df)
    
    # 2. Extract SINGLE CANONICAL LAP
    mx, my = extract_single_lap(mx_all, my_all)
    
    # 3. Calculate Limits for that single lap
    (lx, ly), (rx, ry) = calculate_canonical_boundaries(mx, my, TRACK_WIDTH_METERS)
    
    # 4. Save Canonical Data
    data_path_csv = os.path.join(OUTPUT_DIR, "track_canonical.csv")
    data_path_npz = os.path.join(OUTPUT_DIR, "track_canonical.npz")
    
    pd.DataFrame({
        'center_x': mx, 'center_y': my,
        'left_x': lx,   'left_y': ly,
        'right_x': rx,  'right_y': ry
    }).to_csv(data_path_csv, index=False)
    print(f"Saved Canonical Data (CSV): {data_path_csv}")
    
    np.savez(data_path_npz, center_x=mx, center_y=my, left_x=lx, left_y=ly, right_x=rx, right_y=ry, width=TRACK_WIDTH_METERS)
    print(f"Saved Canonical Data (NPZ): {data_path_npz}")
    
    # 5. Plot V3 (Single Clean Shape)
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12))
    plt.axis('equal')
    
    # Plot Limits
    plt.plot(lx, ly, color='cyan', lw=2, label='Inner/Outer Limit 1')
    plt.plot(rx, ry, color='magenta', lw=2, label='Inner/Outer Limit 2')
    
    # Centerline (Reference)
    plt.plot(mx, my, color='white', lw=1, linestyle='--', alpha=0.3, label='Reference Line')
    
    # Fill Surface
    plt.fill(np.append(lx, rx[::-1]), np.append(ly, ry[::-1]), color='#222222', alpha=0.5, label='Optimal Limit Surface')
    
    plt.title(f"Canonical Track Model (1 Lap) - Width {TRACK_WIDTH_METERS}m", color='white')
    plt.legend(loc='upper right')
    plt.axis('off')
    
    plot_path = os.path.join(OUTPUT_DIR, "track_canonical_v3.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    print(f"Saved Canonical Plot: {plot_path}")
