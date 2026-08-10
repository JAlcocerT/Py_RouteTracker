
import re
import sys

def parse_and_check_low_speeds(file_path):
    print(f"Scanning {file_path} for speeds < 15 km/h...")
    
    # Try different encodings
    encodings = ['utf-16le', 'utf-8', 'latin-1']
    content = None
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except:
            continue
            
    if not content:
        print("Failed to read file.")
        return

    current_sample_time = 0.0
    lines = content.splitlines()
    
    found_low_speed = False
    
    for line in lines:
        # 1. Get Time
        # "Sample Time    : 123.45 s"
        t_match = re.search(r"Sample Time\s+:\s+([\d\.]+)\s+s", line)
        if t_match:
            current_sample_time = float(t_match.group(1))
            
        # 2. Get Speed
        # "GPS Speed      : 10.5" (m/s)
        s_match = re.search(r"GPS Speed\s+:\s+([\d\.]+)", line)
        if s_match:
            speed_ms = float(s_match.group(1))
            speed_kmh = speed_ms * 3.6
            
            if speed_kmh < 15:
                print(f"Time: {current_sample_time:.2f}s | Speed: {speed_kmh:.1f} km/h")

if __name__ == "__main__":
    file_path = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-2b.txt"
    parse_and_check_low_speeds(file_path)
