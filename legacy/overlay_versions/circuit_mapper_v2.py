import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- CONIFGURATION ---
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1c.txt"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
TRACK_WIDTH_METERS = 8.0 
VIDEO_DURATION_SEC = 532.5

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

# --- 2. GEOMETRY UTILS ---
def latlon_to_meters(df):
    lat_mean = df['lat'].mean()
    lat_scale = 111132.954
    lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mean))
    
    # Coordinates in Meters, Centered
    y = (df['lat'] - lat_mean) * lat_scale
    x = (df['lon'] - df['lon'].mean()) * lon_scale
    
    return x.to_numpy(), y.to_numpy()

def calculate_boundaries(x, y, width):
    """
    Computes Left and Right track boundaries based on the driven path.
    """
    # 1. Smooth the path (Simple moving average) to reduce GPS jitter
    # This helps get cleaner normal vectors
    window = 10
    if len(x) > window:
        x_smooth = np.convolve(x, np.ones(window)/window, mode='same')
        y_smooth = np.convolve(y, np.ones(window)/window, mode='same')
    else:
        x_smooth, y_smooth = x, y

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    
    lengths = np.sqrt(dx**2 + dy**2)
    lengths[lengths == 0] = 1.0 # Avoid div/0
    
    ux = dx / lengths
    uy = dy / lengths
    
    # Normal Vector (Rotate 90 deg)
    nx = -uy
    ny = ux
    
    half_width = width / 2.0
    
    # Left Boundary
    lx = x + nx * half_width
    ly = y + ny * half_width
    
    # Right Boundary
    rx = x - nx * half_width
    ry = y - ny * half_width
    
    return (lx, ly), (rx, ry)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    df = parse_lat_lon(FILE_PATH)
    if df.empty: exit(1)
    
    print(f"Processing {len(df)} track points...")
    
    # Convert to Meters
    mx, my = latlon_to_meters(df)
    
    # Calculate Limits
    (lx, ly), (rx, ry) = calculate_boundaries(mx, my, TRACK_WIDTH_METERS)
    
    # --- SAVE DATA (The Answer to "How to save the shape") ---
    # We save as CSV for inspection and NPZ for easy Python loading
    data_path_csv = os.path.join(OUTPUT_DIR, "track_data.csv")
    data_path_npz = os.path.join(OUTPUT_DIR, "track_data.npz")
    
    # CSV
    export_df = pd.DataFrame({
        'center_x': mx, 'center_y': my,
        'left_x': lx,   'left_y': ly,
        'right_x': rx,  'right_y': ry
    })
    export_df.to_csv(data_path_csv, index=False)
    print(f"Saved Track Data (CSV): {data_path_csv}")
    
    # NPZ (Best for loading into optimization scripts)
    np.savez(data_path_npz, 
             center_x=mx, center_y=my, 
             left_x=lx, left_y=ly, 
             right_x=rx, right_y=ry, 
             width=TRACK_WIDTH_METERS)
    print(f"Saved Track Data (NPZ): {data_path_npz}")
    
    # --- PLOT V2 (Outer Limits Only) ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12))
    plt.axis('equal')
    
    # Plot ONLY the boundaries (The "Tube")
    plt.plot(lx, ly, color='cyan', lw=1, label='Left Limit')
    plt.plot(rx, ry, color='magenta', lw=1, label='Right Limit')
    
    # Optional: Fill the "Drivable Surface" for clarity, but minimal
    # We combine them to make a polygon
    # Note: This simple fill works best for closed loops without self-intersection
    plt.fill(np.append(lx, rx[::-1]), np.append(ly, ry[::-1]), color='#222222', alpha=0.3)
    
    plt.title(f"Track Limits Geometry (Width: {TRACK_WIDTH_METERS}m)", color='white')
    plt.legend(loc='upper right')
    
    # Clean output
    plt.axis('off')
    
    plot_path = os.path.join(OUTPUT_DIR, "track_limits_v2.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    print(f"Saved Plot: {plot_path}")
    # plt.show()
