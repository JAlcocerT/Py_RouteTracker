
import re
from collections import Counter

def detect_sample_rate(file_path):
    print(f"Scanning {file_path} for Sample Rate...")
    
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

    lines = content.splitlines()
    sample_times = []
    
    current_time = None
    points_per_sec = Counter()
    
    for line in lines:
        m = re.search(r"Sample Time\s+:\s+([\d\.]+)\s+s", line)
        if m:
            current_time = m.group(1)
            
        if "GPS Speed" in line and current_time:
            points_per_sec[current_time] += 1
            
    # Calculate modes
    counts = list(points_per_sec.values())
    if not counts:
        print("No speed data found.")
        return
        
    avg_rate = sum(counts) / len(counts)
    print(f"Average samples per second block: {avg_rate:.2f}")
    print(f"Min samples: {min(counts)}")
    print(f"Max samples: {max(counts)}")
    print(f"Most common count: {max(set(counts), key=counts.count)}")

if __name__ == "__main__":
    file_path = "/home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/output-kart25dec-2b.txt"
    detect_sample_rate(file_path)
