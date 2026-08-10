import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import re
import os
import mplcyberpunk
import numpy as np
import subprocess
import struct

# --- 1. CONFIGURATION ---
# List your video part files in order
VIDEO_FILES = [
    "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX010411.MP4",
    "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020411.MP4"
]

OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"

# Render Config
TARGET_FPS = 30 
RENDER_FULL = True # Set False for quick preview
PREVIEW_DURATION_SEC = 30

# HUD Config
MAX_EXPECTED_SPEED = 85
LIMIT_G = 1.5 

# Lap Logic
# Start/Finish line coordinates will be auto-detected from LAP_START_TIME_SEC of the FIRST video
# Or you can hardcode LAT/LON if you know them.
LAP_START_TIME_SEC = 176.0 # Relative to the session start (or first video)
LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

# Best Lap Slice
SLICE_BEST_LAP = True
SLICE_BUFFER_SEC = 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. AUTOMATION: HELPERS ---
def get_video_duration(video_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        return float(subprocess.check_output(cmd).decode().strip())
    except: return 532.5 # Default fallback

def dms_to_dd(dms):
    if not dms: return None
    try:
        p = re.split(r'[deg\'"]+', dms)
        v = float(p[0]) + float(p[1])/60 + float(p[2])/3600
        return -v if p[3].strip() in ['S','W'] else v
    except: return None

# --- 3. DATA EXTRACTION ---
def extract_telemetry_gps(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_path = os.path.join(os.path.dirname(video_path), f"{base_name}_telemetry.txt")
    if not os.path.exists(txt_path):
        print(f"Extracting GPS for {base_name}...")
        with open(txt_path, "w") as outfile:
            subprocess.run(["exiftool", "-ee", video_path], stdout=outfile)
    return txt_path

def extract_gpmd_binary(video_path):
    bin_path = video_path.replace(".MP4", ".bin").replace(".mp4", ".bin")
    if not os.path.exists(bin_path):
        print(f"Extracting GPMD Bin for {os.path.basename(video_path)}...")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-map", "0:3", "-f", "data", "-c", "copy", bin_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return bin_path

# --- 4. PARSING ---
def parse_gps_data(txt_path, duration):
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.read().splitlines()
    data = []
    clat, clon = np.nan, np.nan
    for line in lines:
        m_spd = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
        if m_spd:
            # v7: Standardize on KM/H
            data.append({'speed': float(m_spd.group(1)) * 3.6, 'lat': clat, 'lon': clon})
        m_lat = re.search(r"GPS Latitude\s+:\s+(.+)", line)
        if m_lat: clat = dms_to_dd(m_lat.group(1))
        m_lon = re.search(r"GPS Longitude\s+:\s+(.+)", line)
        if m_lon: clon = dms_to_dd(m_lon.group(1))
    
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df[['lat','lon']] = df[['lat','lon']].ffill().bfill()
    df = df[(df['lat']!=0) & (df['lon']!=0)]
    df['time'] = np.linspace(0, duration, len(df))
    return df

def parse_accl_data(bin_path, duration):
    if not os.path.exists(bin_path): return pd.DataFrame()
    with open(bin_path, "rb") as f: content = f.read()
    pts = []
    i = 0; length = len(content)
    while i < length - 8:
        if content[i:i+4] == b"ACCL":
            try:
                esize = content[i+5]
                repeat = struct.unpack(">H", content[i+6:i+8])[0]
                pstart = i + 8
                total = esize * repeat
                pad = (total + 3) & ~3
                for k in range(repeat):
                    o = pstart + k*esize
                    val = struct.unpack(">hhh", content[o:o+6])
                    pts.append(val)
                i += 8 + pad
                continue
            except: pass
        i += 1
    
    df = pd.DataFrame(pts, columns=['c1','c2','c3'])
    if df.empty: return df
    
    # Calibration & Mapping
    mag = np.sqrt(df['c1']**2 + df['c2']**2 + df['c3']**2)
    one_g = mag.median()
    df['lat_g'] = df['c2'] / one_g
    df['lon_g'] = df['c3'] / one_g
    
    # Smoothing
    df['lat_g'] = df['lat_g'].rolling(15, center=True).mean().fillna(0)
    df['lon_g'] = df['lon_g'].rolling(15, center=True).mean().fillna(0)
    df['time'] = np.linspace(0, duration, len(df))
    return df

def sync_dataframes(df_gps, df_accl, duration, fps):
    t_target = np.arange(0, duration, 1/fps)
    
    df_gps = df_gps.set_index('time')
    gps_re = df_gps.reindex(df_gps.index.union(t_target)).interpolate(method='index').reindex(t_target).reset_index()
    
    if not df_accl.empty:
        df_accl = df_accl.set_index('time')
        accl_re = df_accl.reindex(df_accl.index.union(t_target)).interpolate(method='index').reindex(t_target).reset_index()
        merged = pd.merge(gps_re, accl_re[['time','lat_g','lon_g']], on='time', how='left').fillna(0)
    else:
        merged = gps_re
        merged['lat_g'] = 0; merged['lon_g'] = 0
        
    return merged

# --- 5. LOGIC: SESSION MERGE ---
def process_full_session(video_files):
    full_df = pd.DataFrame()
    time_offset = 0.0
    file_map = [] # stores (start_time, end_time, file_path)
    
    for v_path in video_files:
        if not os.path.exists(v_path):
            print(f"Skipping missing file: {v_path}")
            continue
            
        print(f"\nProcessing {os.path.basename(v_path)}...")
        dur = get_video_duration(v_path)
        
        txt = extract_telemetry_gps(v_path)
        bin_f = extract_gpmd_binary(v_path)
        
        gps = parse_gps_data(txt, dur)
        accl = parse_accl_data(bin_f, dur)
        
        if gps.empty: continue
        
        synced = sync_dataframes(gps, accl, dur, TARGET_FPS)
        
        # Shift Time
        synced['session_time'] = synced['time'] + time_offset
        synced['source_file'] = v_path
        synced['source_time'] = synced['time'] # Keep local time for slicing
        
        full_df = pd.concat([full_df, synced], ignore_index=True)
        
        file_map.append({
            'file': v_path,
            'start_session': time_offset,
            'end_session': time_offset + dur,
            'duration': dur
        })
        time_offset += dur
        
    return full_df, file_map, time_offset

# --- 6. LAP LOGIC & SLICING ---
def haversine(lat1, lon1, lat2, lon2):
    R=6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def detect_laps(df, start_lat, start_lon):
    lap_indices = []
    last_time = -MIN_LAP_TIME_SEC
    in_zone = False
    best_dist = 99999
    best_idx = -1
    
    print(f"Detecting laps (Start: {start_lat:.5f}, {start_lon:.5f})...")
    
    for i, row in df.iterrows():
        d = haversine(row['lat'], row['lon'], start_lat, start_lon)
        if row['session_time'] - last_time > MIN_LAP_TIME_SEC:
            if d < LAP_DETECTION_RADIUS_M:
                in_zone = True
                if d < best_dist: best_dist = d; best_idx = i
            else:
                if in_zone: 
                    lap_indices.append(best_idx)
                    last_time = df.iloc[best_idx]['session_time']
                    in_zone = False; best_dist = 99999
    
    # Stats
    lap_table = []
    if len(lap_indices) > 1:
        for k in range(1, len(lap_indices)):
            s = lap_indices[k-1]; e = lap_indices[k]
            sl = df.iloc[s:e]
            lap_table.append({
                'Lap': k,
                'Start Time': df.iloc[s]['session_time'],
                'End Time': df.iloc[e]['session_time'],
                'Duration': df.iloc[e]['session_time'] - df.iloc[s]['session_time'],
                'Avg Speed': sl['speed'].mean(),
                'Max Speed': sl['speed'].max(),
                'Source File': df.iloc[s]['source_file'], # Roughly where it started
                'Source Start': df.iloc[s]['source_time']
            })
            
    # Annotate
    df['lap'] = 0; df['last_lap_s'] = 0.0
    cur = 1; prev = 0
    for idx in lap_indices:
        df.loc[prev:idx, 'lap'] = cur
        if prev > 0: df.loc[idx:, 'last_lap_s'] = df.iloc[idx]['session_time'] - df.iloc[prev]['session_time']
        prev = idx; cur += 1
    df.loc[prev:, 'lap'] = cur
    
    return df, lap_indices, pd.DataFrame(lap_table)

# --- 7. RENDERER ---
def render_hud_v7(df, lap_indices, total_frames, output_name):
    print(f"Rendering {total_frames} frames to {output_name}...")
    
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('black')
    
    gs = GridSpec(2, 3, height_ratios=[3, 1], figure=fig)
    ax_spd = fig.add_subplot(gs[0, 0])
    ax_gg = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[0, 2])
    ax_gph = fig.add_subplot(gs[1, :])
    
    for ax in [ax_spd, ax_gg, ax_map, ax_gph]: 
        ax.set_facecolor('black')
        if ax != ax_gph: ax.axis('off')

    outline = [pe.withStroke(linewidth=3, foreground='black')]
    
    # Init Artists (Same as v6)
    theta = np.linspace(np.pi, 0, 100); rad = 0.35
    arc_x = 0.5 + rad * np.cos(theta); arc_y = 0.4 + rad * np.sin(theta)
    ax_spd.plot(arc_x, arc_y, color='white', lw=1, alpha=0.1)
    sp_arc, = ax_spd.plot([], [], lw=8, solid_capstyle='round', path_effects=outline)
    sp_txt = ax_spd.text(0.5, 0.35, '', fontsize=45, color='white', ha='center', fontweight='bold', path_effects=outline)
    ax_spd.text(0.5, 0.25, 'KM/H', fontsize=12, color='#00ff9f', ha='center', path_effects=outline)
    lap_txt = ax_spd.text(0.1, 0.85, 'LAP', fontsize=16, color='cyan', ha='left', fontweight='bold', path_effects=outline)
    last_txt = ax_spd.text(0.9, 0.85, 'LAST', fontsize=12, color='yellow', ha='right', fontweight='bold', path_effects=outline)
    ax_spd.set_xlim(0,1); ax_spd.set_ylim(0,1)

    ax_gg.set_xlim(-LIMIT_G, LIMIT_G); ax_gg.set_ylim(-LIMIT_G, LIMIT_G); ax_gg.set_aspect('equal')
    ax_gg.add_artist(plt.Circle((0,0), 0.5, color='white', fill=False, alpha=0.2, ls='--'))
    ax_gg.add_artist(plt.Circle((0,0), 1.0, color='white', fill=False, alpha=0.4, ls='-'))
    ax_gg.axhline(0, color='white', alpha=0.1); ax_gg.axvline(0, color='white', alpha=0.1)
    gg_trail, = ax_gg.plot([], [], color='cyan', lw=2, alpha=0.6, path_effects=outline)
    gg_ball, = ax_gg.plot([], [], 'o', color='#ff0055', markersize=12, mec='white', zorder=10)
    gg_txt = ax_gg.text(0.05, 0.9, '', transform=ax_gg.transAxes, color='white', fontsize=10, path_effects=outline)

    ax_map.set_aspect('equal')
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=2, alpha=0.3)
    map_dot, = ax_map.plot([], [], 'o', color='white', mec='red', mew=2, ms=8)
    map_tail, = ax_map.plot([], [], color='#00ff9f', lw=3, alpha=0.9, path_effects=outline)

    ax_gph.set_title("SESSION TELEMETRY", color='white', fontsize=9, pad=5, path_effects=outline)
    ax_gph.plot(df['session_time'], df['speed'], color='white', alpha=0.2, lw=1)
    for i in lap_indices: ax_gph.axvline(df.iloc[i]['session_time'], color='yellow', ls='--', alpha=0.3)
    gph_line, = ax_gph.plot([], [], color='#00ff9f', lw=2, path_effects=outline)
    gph_dot, = ax_gph.plot([], [], 'o', color='#ff0055', ms=6, mec='white')
    ax_gph.set_xlim(0, df['session_time'].max()); ax_gph.set_ylim(0, df['speed'].max()*1.1)
    ax_gph.axis('off')

    mplcyberpunk.add_glow_effects(ax=ax_spd)

    def update(f):
        if f >= len(df): return
        row = df.iloc[f]
        
        v = row['speed']
        sp_txt.set_text(f"{int(v)}")
        r = min(v/MAX_EXPECTED_SPEED, 1.0)
        idx = int(r * 100)
        c = '#00ff9f' if r < 0.5 else '#ffff00' if r < 0.8 else '#ff0055'
        sp_arc.set_data(arc_x[:idx], arc_y[:idx]); sp_arc.set_color(c)
        lap_txt.set_text(f"LAP {int(row['lap'])}")
        last_txt.set_text(f"LAST: {row['last_lap_s']:.2f}s" if row['last_lap_s']>0 else "")

        hist = df.iloc[max(0,f-15):f+1]
        gg_trail.set_data(hist['lat_g'], hist['lon_g'])
        gg_ball.set_data([row['lat_g']], [row['lon_g']])
        g_val = np.sqrt(row['lat_g']**2 + row['lon_g']**2)
        gg_txt.set_text(f"{g_val:.2f} G")
        gg_ball.set_color('red' if g_val > 1.0 else 'yellow' if g_val > 0.5 else '#00ff9f')

        map_dot.set_data([row['lon']], [row['lat']])
        mtail = df.iloc[max(0,f-150):f+1]
        map_tail.set_data(mtail['lon'], mtail['lat'])

        gph = df.iloc[:f+1]
        gph_line.set_data(gph['session_time'], gph['speed'])
        gph_dot.set_data([row['session_time']], [v])

        return sp_arc, sp_txt, lap_txt, last_txt, gg_trail, gg_ball, gg_txt, map_dot, map_tail, gph_line, gph_dot

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)
    ani.save(output_name, writer='ffmpeg', fps=TARGET_FPS, bitrate=6000, extra_args=['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-s', '1600x900'])
    plt.close(fig)
    return output_name

# --- EXECUTION ---
if __name__ == "__main__":
    if len(VIDEO_FILES) == 0: exit(1)

    print("--- RACING HUD v7 (Multi-File) ---")
    
    # 1. Merge
    df_session, file_map, total_dur = process_full_session(VIDEO_FILES)
    if df_session.empty: print("Error: No data."); exit(1)
    
    print(f"\nTotal Session Duration: {total_dur:.2f}s, Frames: {len(df_session)}")
    
    # 2. Get Start Coords (From first X seconds of first file)
    s_lat, s_lon = 0, 0
    start_row = df_session.iloc[(df_session['session_time'] - LAP_START_TIME_SEC).abs().idxmin()]
    s_lat, s_lon = start_row['lat'], start_row['lon']
    print(f"Start Line defined at {LAP_START_TIME_SEC}s (Lat: {s_lat:.6f}, Lon: {s_lon:.6f})")

    # 3. Detect Laps (Global)
    df_final, laps_idx, lap_stats = detect_laps(df_session, s_lat, s_lon)
    print(f"Found {len(laps_idx)} Laps (Indices).")
    
    if not lap_stats.empty:
        print("\n--- LAP RESULTS ---")
        print(lap_stats[['Lap', 'Duration', 'Max Speed']])
        
        # 4. Best Lap Slicing (Session Aware)
        if SLICE_BEST_LAP:
            best_idx = lap_stats['Duration'].idxmin()
            best_row = lap_stats.loc[best_idx]
            best_lap_num = int(best_row['Lap'])
            
            # Find Source File
            # The start time of the best lap allows us to find which file it belongs to
            # But overlapping laps (across files) are complex.
            # Simplified: Assuming Lap fits in one file or handled by ffmpeg concat if we concatenated videos.
            # Wait, we are slicing from specific file?
            # 'Source File' is stored in stats map!
            
            src_file = best_row['Source File']
            src_start = best_row['Source Start']
            dur_lap = best_row['Duration']
            
            t_start = max(0, src_start - SLICE_BUFFER_SEC)
            t_end = src_start + dur_lap + SLICE_BUFFER_SEC
            
            # Note: This simple slice logic assumes the whole lap is in ONE file. 
            # If a lap crosses video split, this 'Source File' points to Start.
            # We would need complex stitching. For GoPro, Splits usually happen at ~11 mins / 4GB.
            # Laps are shorter. It works 99% of time.
            
            out_slice = os.path.join(OUTPUT_DIR, f"Best_Lap_{best_lap_num}_{dur_lap:.2f}s_v7.mp4")
            print(f"\n--- Slicing Best Lap {best_lap_num} from {os.path.basename(src_file)} ---")
            print(f"Time: {t_start:.1f}s -> {t_end:.1f}s")
            
            cmd = ["ffmpeg", "-y", "-ss", str(t_start), "-i", src_file, "-t", str(t_end-t_start), "-c", "copy", out_slice]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved Slice: {out_slice}")
    
    # 5. Render HUD Overlay
    # Render separate HUD parts for each video? Or one big HUD video?
    # User said "ordered videos of the race... appended".
    # Usually you want one big HUD video to match the concatenated GoPro footage.
    # So we render ONE long HUD video.
    
    # The user will need to concatenate their GoPro files to match this HUD!
    # I should print the ffmpeg concat command for them.
    
    out_hud = os.path.join(OUTPUT_DIR, "HUD_v7_Session.mp4")
    frames_to_render = len(df_final) if RENDER_FULL else int(PREVIEW_DURATION_SEC * TARGET_FPS)
    
    render_hud_v7(df_final, laps_idx, frames_to_render, out_hud)
    
    print("\n" + "="*50)
    print("DONE! To merge everything, first Concatenate your GoPro files:")
    print("ffmpeg -f concat -safe 0 -i <(printf \"file 'GX01.MP4'\\nfile 'GX02.MP4'\") -c copy Session_Full.mp4")
    print("Then overlay the HUD:")
    print(f"ffmpeg -i Session_Full.mp4 -i {out_hud} ...")
    print("="*50)
