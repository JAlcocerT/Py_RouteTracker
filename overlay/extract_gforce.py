import struct
import os
import pandas as pd
import numpy as np
import subprocess
import json

# CONFIGURATION
VIDEO_PATH = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4"
OUTPUT_CSV = "gforce_data.csv"

# CONSTANTS (GoPro SCALING)
# ACCL is usually packed as signed 16-bit integers or floats depending on camera
# HERO 9/10/11 usually use standard scaling.
# We will use a library approach if available, but let's try a simple binary dump trick first.

# STRATEGY: 
# Using ffmpeg to dump the specific 'gpmd' stream to a raw binary file,
# then we parse that binary file in Python.
# This avoids writing a full MP4 atom parser.

def extract_gpmd_stream(video_path):
    print(f"Extracting raw GPMD stream from {video_path}...")
    bin_path = video_path.replace(".MP4", ".bin").replace(".mp4", ".bin")
    
    # Map the GPMD stream (Stream #0:3 usually)
    # Check ffprobe first to find the stream index with 'gpmd'
    try:
        cmd_probe = ["ffprobe", "-v", "error", "-select_streams", "d", "-show_entries", "stream=index:stream_tags=handler_name", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, text=True)
        # Parse output to find index. Usually "3,GoPro MET"
        lines = result.stdout.strip().splitlines()
        stream_index = -1
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2 and "GoPro MET" in parts[1]:
                stream_index = int(parts[0])
                break
        
        if stream_index == -1:
            # Fallback: Tag binary data often at map 0:3
            stream_index = 3 
            
        print(f"Found GPMD at stream index {stream_index}")
        
        # Dump to binary
        cmd_dump = [
            "ffmpeg", "-y", "-i", video_path, 
            "-map", f"0:{stream_index}", 
            "-f", "data", # Dump raw data
            "-c", "copy", 
            bin_path
        ]
        subprocess.run(cmd_dump, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
            print(f"Raw binary extracted: {bin_path} ({os.path.getsize(bin_path)} bytes)")
            return bin_path
        else:
            print("Failed to extract binary stream.")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_gpmd_binary(bin_path):
    print("Parsing binary GPMD data...")
    data_points = []
    
    with open(bin_path, "rb") as f:
        content = f.read()
        
    # GPMD Format: [FourCC (4)][Type (1)][Size (1)][Structure Size (2)][Repeat (2)][Data...]
    # We scan for 'ACCL' tag
    
    i = 0
    length = len(content)
    
    # Sticky scalers
    scale = 1.0 # Default
    
    while i < length - 8:
        # Check if we are at a valid KLV start
        # Key (4 bytes)
        key = content[i:i+4].decode('ascii', errors='ignore')
        
        # This is a naive seek. The structure is hierarchical (DEVC -> DVID -> STRM -> ACCL)
        # But scanning for 'ACCL' usually works for quick extraction.
        
        if key == "SCAL":
            # Scaling factor for next data
            try:
                # Type, Size, etc.
                # Assuming standard KLV
                # Skip 4 bytes key.
                # Type (1) Size (1) Count (2)
                type_byte = content[i+4]
                size_byte = content[i+5]
                count = struct.unpack(">H", content[i+6:i+8])[0]
                
                # Payload
                payload_len = size_byte * count
                # Let's verify alignment (usually 4-byte aligned?)
                
                # If valid SCAL, update scale
                # Usually signed int 16 or 32
                pass # Implementing full parser is complex, skipping strict SCAL for now
            except: pass

        if key == "ACCL":
            try:
                # Parse Header
                # i+0: ACCL
                # i+4: Type (1 byte) -> 's' (short), 'f' (float), 'L' (long)
                # i+5: Element Size (1 byte) -> usually 6 (3 shorts) or 12 (3 floats)
                # i+6: Repeat Count (2 bytes) -> How many samples in this payload
                
                dtype_char = chr(content[i+4])
                elem_size = content[i+5]
                repeat = struct.unpack(">H", content[i+6:i+8])[0]
                
                payload_start = i + 8
                total_bytes = elem_size * repeat
                
                # Move align
                padded_bytes = (total_bytes + 3) & ~3
                
                # Parse Samples
                samples = []
                for k in range(repeat):
                    offset = payload_start + (k * elem_size)
                    
                    if dtype_char == 's': # Signed Short (standard for older Heros)
                        # Usually packed y,x,z ? or z,x,y?
                        # Scaler is needed. Usually SCAL was immediately before.
                        # Approx scale for MP4: Often / 1 (m/s2)? Or / some factor.
                        # Let's assume raw first.
                        val_raw = struct.unpack(">hhh", content[offset:offset+6])
                        samples.append(val_raw)
                        
                    elif dtype_char == 'l': # Signed Long
                        val_raw = struct.unpack(">lll", content[offset:offset+12])
                        samples.append(val_raw)
                        
                # Add to data
                for s in samples:
                    # GPMD Standard: Y, X, Z? 
                    # Usually: Z (Vertical), X (Lat), Y (Long)?
                    # We will dump as is (1,2,3)
                    data_points.append({'c1': s[0], 'c2': s[1], 'c3': s[2]})
                
                # Advance
                # i += 8 + padded_bytes (KLV header + Payload)
                # But since we are brute scanning, let's just jump
                i += 8 + padded_bytes
                continue
                
            except Exception as e:
                # Parse error, just move forward 1 byte
                pass
        
        i += 1
        
    df = pd.DataFrame(data_points)
    
    # Auto-Calibration Estimate
    # Gravity is ~9.8 on one axis usually (when static) or ~0 if properly zeroed?
    # GoPro usually stores raw.
    # If values are huge (>1000), likely Short needing divider.
    # GoPro Scaler is often specific.
    
    if not df.empty:
        max_val = df.abs().max().max()
        if max_val > 500: # Heuristic for 'Integer format'
            # Check range. 
            # Often scaler is around 400-500 for 1G?
            pass
            
    return df

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print("Video not found.")
        exit(1)
        
    bin_file = extract_gpmd_stream(VIDEO_PATH)
    if bin_file:
        df = parse_gpmd_binary(bin_file)
        if not df.empty:
            print(f"Parsed {len(df)} ACCL samples.")
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Saved to {OUTPUT_CSV}")
            
            # Simple Plot
            try:
                import matplotlib.pyplot as plt
                import mplcyberpunk
                plt.style.use("cyberpunk")
                plt.figure(figsize=(12, 6))
                # Plot subsets to see
                subset = df.head(1000)
                plt.plot(subset['c1'], label='Ch 1', alpha=0.6)
                plt.plot(subset['c2'], label='Ch 2', alpha=0.6)
                plt.plot(subset['c3'], label='Ch 3', alpha=0.6)
                plt.legend()
                plt.title("Raw GPMD ACCL Data (First 1000 pts)")
                plt.savefig("gforce_debug.png")
                print("Plot saved: gforce_debug.png")
            except: pass
        else:
            print("No ACCL data found in binary stream.")
