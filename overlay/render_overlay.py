import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
import os
import mplcyberpunk
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FILE = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-1b.txt" 
OUTPUT_DIR = "/home/jalcocert/Desktop/Py_RouteTracker/overlay"
SAMPLE_FPS = 18  # Approximate GPS sampling rate from file
#RENDER_SECONDS = 5 # Used only if RENDER_FULL is False
RENDER_FULL = True # Set to True for the full video!

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. PARSING LOGIC ---
def parse_gopro_txt(file_path):
    data = []
    current_sample_time = 0.0
    
    # ExifTool often outputs UTF-16LE or UTF-8 depending on the OS/Command.
    # We try both or use chardet if needed, but here we'll try UTF-16LE first as it was detected.
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                if "Sample Time" in content:
                    print(f"Successfully opened with {enc}")
                    lines = content.splitlines()
                    for line in lines:
                        sample_match = re.search(r"Sample Time\s+:\s+([\d\.]+)\s+s", line)
                        if sample_match:
                            current_sample_time = float(sample_match.group(1))
                        
                        speed_match = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
                        if speed_match:
                            speed = float(speed_match.group(1))
                            micro_offset = len([d for d in data if d['sample_time'] == current_sample_time]) * (1.0/SAMPLE_FPS)
                            data.append({
                                'time': current_sample_time + micro_offset,
                                'sample_time': current_sample_time,
                                'speed_kmh': speed * 1
                            })
                    return pd.DataFrame(data)
        except Exception:
            continue
    
    return pd.DataFrame(data)

# --- 3. RENDER LOGIC (Matplotlib) ---
def render_speedometer(df):
    plt.style.use('cyberpunk')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 + 0.4 * np.cos(theta)
    circle_y = 0.5 + 0.4 * np.sin(theta)
    circle, = ax.plot(circle_x, circle_y, color='#00ff9f', lw=4, alpha=0.8)
    
    speed_text = ax.text(0.5, 0.45, '0', fontsize=50, color='white', ha='center', fontweight='bold')
    unit_text = ax.text(0.5, 0.3, 'KM/H', fontsize=12, color='#00ff9f', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def update(frame):
        if frame < len(df):
            val = df.iloc[frame]['speed_kmh']
            speed_text.set_text(f"{int(val)}")
        return speed_text, circle

    num_frames = len(df) if RENDER_FULL else int(RENDER_SECONDS * SAMPLE_FPS)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    mplcyberpunk.add_glow_effects()
    
    save_path = os.path.join(OUTPUT_DIR, "speedometer_overlay.mp4")
    print(f"Rendering {num_frames} frames to {save_path}...")
    
    try:
        # We try to use a black background for easiest overlaying later
        ani.save(save_path, writer='ffmpeg', fps=SAMPLE_FPS, bitrate=2000, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    except Exception as e:
        print(f"Error saving MP4: {e}. Falling back to GIF.")
        save_path = save_path.replace(".mp4", ".gif")
        ani.save(save_path, writer='pillow', fps=SAMPLE_FPS)
        
    plt.close()
    return save_path

# --- RUN ---
print("Extracting telemetry data...")
df_telemetry = parse_gopro_txt(INPUT_FILE)
print(f"Extracted {len(df_telemetry)} points.")

if len(df_telemetry) > 0:
    render_speedometer(df_telemetry)
    print("\nDONE! You can find the overlay in the /overlay folder.")
else:
    print("No data found. Check the file path and encoding.")