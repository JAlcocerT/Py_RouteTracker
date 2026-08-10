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
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020411.MP4"
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"

# Render Config
TARGET_FPS = 30 # Smoother for G-Force
RENDER_SECONDS = 30
RENDER_FULL = True

# HUD Config
MAX_EXPECTED_SPEED = 85
LIMIT_G = 1.5 

# Lap Logic
LAP_START_TIME_SEC = 40 #GX020411 #176.0 GX010411
LAP_DETECTION_RADIUS_M = 15.0
MIN_LAP_TIME_SEC = 30.0

# Best Lap Slicing (From v5)
SLICE_BEST_LAP = True
SLICE_BUFFER_SEC = 5.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. AUTOMATION: HELPERS ---
def get_video_duration(video_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        return float(subprocess.check_output(cmd).decode().strip())
    except: return 532.5

# --- 3. GPS EXTRACTION & PARSING (From V5) ---
def extract_telemetry_gps(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_path = os.path.join(os.path.dirname(video_path), f"{base_name}_telemetry.txt")
    if not os.path.exists(txt_path):
        print("Extracting GPS via exiftool...")
        with open(txt_path, "w") as outfile:
            subprocess.run(["exiftool", "-ee", video_path], stdout=outfile)
    return txt_path

def dms_to_dd(dms):
    if not dms: return None
    try:
        p = re.split(r'[deg\'"]+', dms)
        v = float(p[0]) + float(p[1])/60 + float(p[2])/3600
        return -v if p[3].strip() in ['S','W'] else v
    except: return None

def parse_gps_data(txt_path, duration):
    print("Parsing GPS...")
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.read().splitlines()
    data = []
    clat, clon = np.nan, np.nan
    for line in lines:
        m_spd = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
        if m_spd:
            # V5 Fix: * 3.6 for km/h
            data.append({'speed': float(m_spd.group(1)) * 3.6, 'lat': clat, 'lon': clon})
        m_lat = re.search(r"GPS Latitude\s+:\s+(.+)", line)
        if m_lat: clat = dms_to_dd(m_lat.group(1))
        m_lon = re.search(r"GPS Longitude\s+:\s+(.+)", line)
        if m_lon: clon = dms_to_dd(m_lon.group(1))
    
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df[['lat','lon']] = df[['lat','lon']].ffill().bfill()
    df = df[(df['lat']!=0) & (df['lon']!=0)]
    
    # Time Mapping
    df['time'] = np.linspace(0, duration, len(df))
    return df

# --- 4. GSERVER EXTRACTION & PARSING (From V1) ---
def extract_gpmd_binary(video_path):
    bin_path = video_path.replace(".MP4", ".bin").replace(".mp4", ".bin")
    if not os.path.exists(bin_path):
        print("Extracting Raw GPMD Binary...")
        # Find stream 3 usually
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-map", "0:3", "-f", "data", "-c", "copy", bin_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return bin_path

def parse_accl_data(bin_path, duration):
    print("Parsing ACCL Binary...")
    if not os.path.exists(bin_path): return pd.DataFrame()
    with open(bin_path, "rb") as f: content = f.read()
    
    pts = []
    i = 0; length = len(content)
    while i < length - 8:
        if content[i:i+4] == b"ACCL":
            try:
                # Type(1), Size(1), Repeat(2)
                esize = content[i+5]
                repeat = struct.unpack(">H", content[i+6:i+8])[0]
                pstart = i + 8
                total = esize * repeat
                pad = (total + 3) & ~3
                
                for k in range(repeat):
                    o = pstart + k*esize
                    # Signed Short
                    val = struct.unpack(">hhh", content[o:o+6])
                    pts.append(val)
                i += 8 + pad
                continue
            except: pass
        i += 1
    
    df = pd.DataFrame(pts, columns=['c1','c2','c3'])
    if df.empty: return df
    
    # AUTO CALIBRATE 1G
    mag = np.sqrt(df['c1']**2 + df['c2']**2 + df['c3']**2)
    one_g = mag.median()
    print(f"ACCL Calibration: 1G = {one_g:.1f}")
    
    # Axis Mapping: C1=Vert, C2=Lat, C3=Lon (Approx)
    df['lat_g'] = df['c2'] / one_g
    df['lon_g'] = df['c3'] / one_g
    
    # Smoothing
    df['lat_g'] = df['lat_g'].rolling(15, center=True).mean().fillna(0)
    df['lon_g'] = df['lon_g'].rolling(15, center=True).mean().fillna(0)
    
    df['time'] = np.linspace(0, duration, len(df))
    return df

# --- 5. SYNCHRONIZATION ---
def sync_dataframes(df_gps, df_accl, duration, fps):
    print("Synchronizing Streams...")
    t_target = np.arange(0, duration, 1/fps)
    
    # Resample GPS
    df_gps = df_gps.set_index('time')
    gps_re = df_gps.reindex(df_gps.index.union(t_target)).interpolate(method='index').reindex(t_target).reset_index()
    gps_re = gps_re.rename(columns={'index': 'time'})
    
    # Resample ACCL
    df_accl = df_accl.set_index('time')
    accl_re = df_accl.reindex(df_accl.index.union(t_target)).interpolate(method='index').reindex(t_target).reset_index()
    
    # Merge
    merged = pd.merge(gps_re, accl_re[['time','lat_g','lon_g']], on='time', how='left').fillna(0)
    return merged

# --- 6. LAP LOGIC (Enhanced with Stats) ---
def haversine(lat1, lon1, lat2, lon2):
    R=6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def detect_laps(df):
    start_row = df.iloc[(df['time']-LAP_START_TIME_SEC).abs().idxmin()]
    slat, slon = start_row['lat'], start_row['lon']
    
    # Find Indices
    laps_idx = []
    last_time = -MIN_LAP_TIME_SEC
    in_zone = False
    best_dist = 99999
    best_idx = -1
    
    for i, row in df.iterrows():
        d = haversine(row['lat'], row['lon'], slat, slon)
        if row['time'] - last_time > MIN_LAP_TIME_SEC:
            if d < LAP_DETECTION_RADIUS_M:
                in_zone = True
                if d < best_dist: best_dist = d; best_idx = i
            else:
                if in_zone: # Exited zone
                    laps_idx.append(best_idx)
                    last_time = df.iloc[best_idx]['time']
                    in_zone = False; best_dist = 99999
    
    # Build Stats Table (v5 Logic)
    lap_table = []
    if len(laps_idx) > 1:
        for k in range(1, len(laps_idx)):
            start_idx = laps_idx[k-1]
            end_idx = laps_idx[k]
            row_start = df.iloc[start_idx]
            row_end = df.iloc[end_idx]
            
            lap_slice = df.iloc[start_idx:end_idx]
            avg_spd = lap_slice['speed'].mean() if 'speed' in lap_slice else 0
            max_spd = lap_slice['speed'].max() if 'speed' in lap_slice else 0
            
            lap_table.append({
                'Lap': k,
                'Start Time': row_start['time'],
                'End Time': row_end['time'],
                'Duration': row_end['time'] - row_start['time'],
                'Avg Speed': avg_spd,
                'Max Speed': max_spd
            })
            
    # Annotate DF for HUD
    df['lap'] = 0; df['last_lap_s'] = 0.0
    cur = 1; prev = 0
    for idx in laps_idx:
        df.loc[prev:idx, 'lap'] = cur
        if prev > 0: df.loc[idx:, 'last_lap_s'] = df.iloc[idx]['time'] - df.iloc[prev]['time']
        prev = idx; cur += 1
    df.loc[prev:, 'lap'] = cur
    
    return df, laps_idx, pd.DataFrame(lap_table)

# --- 7. RENDERER ---
def render_hud_v6(df, lap_indices):
    print(f"Rendering {len(df)} frames...")
    
    plt.style.use('cyberpunk')
    fig = plt.figure(figsize=(16, 9), dpi=100) # 16:9 Full Canvas
    fig.patch.set_facecolor('black')
    
    # Layout:
    # Top Left: Speedometer
    # Top Center: Friction Circle
    # Top Right: Track Map
    # Bottom: Speed Graph
    
    gs = GridSpec(2, 3, height_ratios=[3, 1], figure=fig)
    ax_spd = fig.add_subplot(gs[0, 0])
    ax_gg = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[0, 2])
    ax_gph = fig.add_subplot(gs[1, :])
    
    for ax in [ax_spd, ax_gg, ax_map, ax_gph]: 
        ax.set_facecolor('black')
        if ax != ax_gph: ax.axis('off')

    outline = [pe.withStroke(linewidth=3, foreground='black')]
    
    # 1. SPEEDOMETER
    theta = np.linspace(np.pi, 0, 100)
    rad = 0.35
    arc_x = 0.5 + rad * np.cos(theta)
    arc_y = 0.4 + rad * np.sin(theta)
    ax_spd.plot(arc_x, arc_y, color='white', lw=1, alpha=0.1)
    sp_arc, = ax_spd.plot([], [], lw=8, solid_capstyle='round', path_effects=outline)
    sp_txt = ax_spd.text(0.5, 0.35, '', fontsize=45, color='white', ha='center', fontweight='bold', path_effects=outline)
    ax_spd.text(0.5, 0.25, 'KM/H', fontsize=12, color='#00ff9f', ha='center', path_effects=outline)
    lap_txt = ax_spd.text(0.1, 0.85, 'LAP', fontsize=16, color='cyan', ha='left', fontweight='bold', path_effects=outline)
    last_txt = ax_spd.text(0.9, 0.85, 'LAST', fontsize=12, color='yellow', ha='right', fontweight='bold', path_effects=outline)
    ax_spd.set_xlim(0,1); ax_spd.set_ylim(0,1)

    # 2. FRICTION CIRCLE
    ax_gg.set_xlim(-LIMIT_G, LIMIT_G); ax_gg.set_ylim(-LIMIT_G, LIMIT_G); ax_gg.set_aspect('equal')
    ax_gg.add_artist(plt.Circle((0,0), 0.5, color='white', fill=False, alpha=0.2, ls='--'))
    ax_gg.add_artist(plt.Circle((0,0), 1.0, color='white', fill=False, alpha=0.4, ls='-'))
    ax_gg.axhline(0, color='white', alpha=0.1); ax_gg.axvline(0, color='white', alpha=0.1)
    gg_trail, = ax_gg.plot([], [], color='cyan', lw=2, alpha=0.6, path_effects=outline)
    gg_ball, = ax_gg.plot([], [], 'o', color='#ff0055', markersize=12, markeredgecolor='white', zorder=10)
    gg_txt = ax_gg.text(0.05, 0.9, '', transform=ax_gg.transAxes, color='white', fontsize=10, path_effects=outline)

    # 3. TRACK MAP
    ax_map.set_aspect('equal')
    ax_map.plot(df['lon'], df['lat'], color='cyan', lw=2, alpha=0.3)
    map_dot, = ax_map.plot([], [], 'o', color='white', markeredgecolor='red', markeredgewidth=2, markersize=8)
    map_tail, = ax_map.plot([], [], color='#00ff9f', lw=3, alpha=0.9, path_effects=outline)

    # 4. SPEED GRAPH
    ax_gph.set_title("SPEED TELEMETRY", color='white', fontsize=9, pad=5, path_effects=outline)
    ax_gph.plot(df['time'], df['speed'], color='white', alpha=0.2, lw=1)
    for i in lap_indices: 
        ax_gph.axvline(df.iloc[i]['time'], color='yellow', ls='--', alpha=0.3)
    gph_line, = ax_gph.plot([], [], color='#00ff9f', lw=2, path_effects=outline)
    gph_dot, = ax_gph.plot([], [], 'o', color='#ff0055', ms=6, mec='white')
    ax_gph.set_xlim(0, df['time'].max()); ax_gph.set_ylim(0, df['speed'].max()*1.1)
    ax_gph.axis('off') # Cleaner look

    # ANIMATION
    def update(f):
        if f >= len(df): return
        row = df.iloc[f]
        
        # Speed
        v = row['speed']
        sp_txt.set_text(f"{int(v)}")
        r = min(v/MAX_EXPECTED_SPEED, 1.0)
        idx = int(r * 100)
        c = '#00ff9f' if r < 0.5 else '#ffff00' if r < 0.8 else '#ff0055'
        sp_arc.set_data(arc_x[:idx], arc_y[:idx])
        sp_arc.set_color(c)
        lap_txt.set_text(f"LAP {int(row['lap'])}")
        last_txt.set_text(f"LAST: {row['last_lap_s']:.2f}s" if row['last_lap_s']>0 else "")

        # GG
        lat, lon = row['lat_g'], row['lon_g']
        hist = df.iloc[max(0,f-15):f+1]
        gg_trail.set_data(hist['lat_g'], hist['lon_g'])
        gg_ball.set_data([lat], [lon])
        g_val = np.sqrt(lat**2 + lon**2)
        gg_txt.set_text(f"{g_val:.2f} G")
        gg_ball.set_color('red' if g_val > 1.0 else 'yellow' if g_val > 0.5 else '#00ff9f')

        # Map
        map_dot.set_data([row['lon']], [row['lat']])
        mtail = df.iloc[max(0,f-150):f+1]
        map_tail.set_data(mtail['lon'], mtail['lat'])

        # Graph
        gph = df.iloc[:f+1]
        gph_line.set_data(gph['time'], gph['speed'])
        gph_dot.set_data([row['time']], [v])

        return sp_arc, sp_txt, lap_txt, last_txt, gg_trail, gg_ball, gg_txt, map_dot, map_tail, gph_line, gph_dot

    mplcyberpunk.add_glow_effects(ax=ax_spd)
    
    frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * TARGET_FPS)
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    
    sname = f"HUD_v6_{os.path.basename(VIDEO_PATH)[:-4]}.mp4"
    spath = os.path.join(OUTPUT_DIR, sname)
    print(f"Saving to {spath}...")
    ani.save(spath, writer='ffmpeg', fps=TARGET_FPS, bitrate=6000, extra_args=['-c:v', 'libx264', '-pix_fmt', 'yuv420p'])
    return spath

# --- EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH): exit(1)
    
    # 1. Automate Infos
    dur = get_video_duration(VIDEO_PATH)
    print(f"Duration: {dur:.2f}s")
    
    # 2. Get Data
    txt = extract_telemetry_gps(VIDEO_PATH)
    bin_f = extract_gpmd_binary(VIDEO_PATH)
    
    df_gps = parse_gps_data(txt, dur)
    df_accl = parse_accl_data(bin_f, dur)
    
    if df_gps.empty: print("No GPS data!"); exit(1)
    
    # 3. Sync
    if not df_accl.empty:
        df_merged = sync_dataframes(df_gps, df_accl, dur, TARGET_FPS)
    else:
        print("No ACCL data, using GPS only.")
        df_merged = df_gps
        df_merged['lat_g'] = 0; df_merged['lon_g'] = 0
    
    # 4. Laps
    df_final, laps_idx, lap_stats = detect_laps(df_merged)
    print(f"Found {len(laps_idx)} Laps (Indices).")
    
    if not lap_stats.empty:
        print("\n--- LAP RESULTS ---")
        print(lap_stats[['Lap', 'Duration', 'Max Speed']])
        
        # Youtube Chapters
        print("\n--- Youtube Chapters ---")
        print("00:00 - Intro")
        best_lap_idx = lap_stats['Duration'].idxmin()
        best_lap_num = lap_stats.loc[best_lap_idx, 'Lap']
        
        for i, row in lap_stats.iterrows():
            st = int(row['Start Time']); m, s = divmod(st, 60)
            mark = "🔥 BEST LAP" if row['Lap'] == best_lap_num else ""
            print(f"{m:02d}:{s:02d} - Lap {int(row['Lap'])} ({row['Duration']:.2f}s) {mark}")
            
        # 5a. Generate Static Analysis Plot (From v5)
        try:
            plt.style.use("cyberpunk")
        except: plt.style.use('dark_background')
        
        fig_stat, ax_stat = plt.subplots(figsize=(14, 7))
        ax_stat.plot(df_final['time'], df_final['speed'], color='#00ff9f', lw=1.5, label='Speed')
        ax_stat.axvline(x=LAP_START_TIME_SEC, color='yellow', linestyle=':', alpha=0.8, label='Start')
        
        for lx in laps_idx:
            ax_stat.axvline(x=df_final.iloc[lx]['time'], color='white', linestyle='--', alpha=0.5)
            
        for k in range(1, len(laps_idx)):
            s_idx = laps_idx[k-1]; e_idx = laps_idx[k]
            sl = df_final.iloc[s_idx:e_idx]
            mx_i = sl['speed'].idxmax(); mx_v = sl['speed'].max()
            mn_i = sl['speed'].idxmin(); mn_v = sl['speed'].min()
            
            # Annotations
            ax_stat.annotate(f"{mx_v:.1f}", (df_final.loc[mx_i,'time'], mx_v), xytext=(0,10), 
                             textcoords='offset points', color='#ff0055', ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', color='#ff0055'))
            ax_stat.annotate(f"{mn_v:.1f}", (df_final.loc[mn_i,'time'], mn_v), xytext=(0,-15), 
                             textcoords='offset points', color='cyan', ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', color='cyan'))
            
            # Label
            mid = (df_final.iloc[s_idx]['time'] + df_final.iloc[e_idx]['time'])/2
            dur_val = (df_final.iloc[e_idx]['time'] - df_final.iloc[s_idx]['time'])
            ax_stat.text(mid, mx_v*1.05, f"L{k}\n{dur_val:.1f}s", color='white', ha='center', 
                         fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
                         
        pname = os.path.join(OUTPUT_DIR, f"lap_analysis_{os.path.basename(VIDEO_PATH)[:-4]}.png")
        fig_stat.savefig(pname, dpi=100)
        plt.close(fig_stat)
        print(f"Saved Plot: {pname}")

        # 5b. Best Lap Slicing
        if SLICE_BEST_LAP:
            best_row = lap_stats.loc[best_lap_idx]
            t_start = max(0, best_row['Start Time'] - SLICE_BUFFER_SEC)
            t_end = min(dur, best_row['End Time'] + SLICE_BUFFER_SEC)
            
            out_slice = os.path.join(OUTPUT_DIR, f"Best_Lap_{int(best_lap_num)}_{best_row['Duration']:.2f}s.mp4")
            print(f"\n--- Slicing Best Lap {int(best_lap_num)}: {t_start:.1f}s -> {t_end:.1f}s ---")
            
            cmd = ["ffmpeg", "-y", "-ss", str(t_start), "-i", VIDEO_PATH, "-t", str(t_end-t_start), "-c", "copy", out_slice]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Saved Slice: {out_slice}")

    # 6. Render HUD
    out = render_hud_v6(df_final, laps_idx)
    print(f"DONE: {out}")
