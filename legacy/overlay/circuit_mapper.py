import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math

try:
    import folium
except ImportError:
    folium = None
    print("Warning: 'folium' not found. Leaflet map will not be generated. Install with: pip install folium")

# --- CONFIGURATION ---
FILE_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
TRACK_WIDTH_METERS = 8.0  # Estimated width of the karting track
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
                # We extract coordinates whenever they appear
                # The file structure often has Lat/Lon lines separate from speed data
                # We just want the raw coordinate stream
                lat_m = re.search(r"GPS Latitude\s+:\s+(.+)", line)
                if lat_m: cur_lat = dms_to_dd(lat_m.group(1))
                
                lon_m = re.search(r"GPS Longitude\s+:\s+(.+)", line)
                if lon_m: cur_lon = dms_to_dd(lon_m.group(1))
                
                if not np.isnan(cur_lat) and not np.isnan(cur_lon):
                    # Dedup: Only add if changed significantly (optional, but good for noise)
                    if not data or (data[-1][0] != cur_lat or data[-1][1] != cur_lon):
                        data.append([cur_lat, cur_lon])
            
            if data:
                return pd.DataFrame(data, columns=['lat', 'lon'])
        except: continue
    
    return pd.DataFrame()

# --- 2. GEOMETRY UTILS ---
def latlon_to_meters(df):
    """
    Project Lat/Lon to local Metric approximation (Flat Earth near coords).
    1 deg Lat ~= 111132.954 meters
    1 deg Lon ~= 111132.954 * cos(lat) meters
    """
    lat_mean = df['lat'].mean()
    lat_scale = 111132.954
    lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mean))
    
    # Center to 0,0 for easier plotting
    y = (df['lat'] - lat_mean) * lat_scale
    x = (df['lon'] - df['lon'].mean()) * lon_scale
    
    return x, y

def calculate_track_limits(x, y, width):
    """
    Calculate Left and Right boundaries of the track.
    vectors: P[i], Normal[i]
    """
    # Calculate gradients (tangent vectors)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Normalize tangents to get unit vectors
    lengths = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
    lengths[lengths == 0] = 1.0
    
    ux = dx / lengths
    uy = dy / lengths
    
    # Normal vector is (-y, x) of the tangent for 90 degree rotation
    nx = -uy
    ny = ux
    
    # Offset points
    half_width = width / 2.0
    
    # Inner/Outer depends on direction (CW vs CCW), we just call them Left/Right
    x_left = x + nx * half_width
    y_left = y + ny * half_width
    
    x_right = x - nx * half_width
    y_right = y - ny * half_width
    
    return (x_left, y_left), (x_right, y_right)

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    df = parse_lat_lon(FILE_PATH)
    
    if df.empty:
        print("Error: No GPS data found.")
        exit(1)
        
    print(f"Extracted {len(df)} track points.")
    
    # A. LEAFLET MAP
    if folium:
        print("Generating Leaflet Map...")
        center_lat, center_lon = df['lat'].mean(), df['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')
        
        # Draw Track Line
        points = df[['lat', 'lon']].values.tolist()
        folium.PolyLine(points, color="#00ff9f", weight=5, opacity=0.8).add_to(m)
        
        # Fit bounds
        m.fit_bounds([
            [df['lat'].min(), df['lon'].min()],
            [df['lat'].max(), df['lon'].max()]
        ])
        
        map_path = os.path.join(OUTPUT_DIR, "circuit_map.html")
        m.save(map_path)
        print(f"Saved: {map_path}")
    
    # B. MATPLOTLIB LIMITS PLOT
    print("Calculating approximate circuit limits...")
    mx, my = latlon_to_meters(df)
    (lx, ly), (rx, ry) = calculate_track_limits(mx, my, TRACK_WIDTH_METERS)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    
    # Plot Boundaries
    plt.fill(np.append(lx, rx[::-1]), np.append(ly, ry[::-1]), color='#333333', alpha=0.5, label='Track Surface')
    plt.plot(lx, ly, color='white', lw=1, alpha=0.5, linestyle='--', label='Limits')
    plt.plot(rx, ry, color='white', lw=1, alpha=0.5, linestyle='--')
    
    # Plot Center Line
    plt.plot(mx, my, color='#00ff9f', lw=2, label='Driven Path')
    
    plt.title(f"Approximated Track Shape (Width: {TRACK_WIDTH_METERS}m)", color='white')
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    # Remove axes for clean shape view
    plt.axis('off')
    
    plot_path = os.path.join(OUTPUT_DIR, "circuit_shape.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {plot_path}")
    plt.show() # Optional if interactive
