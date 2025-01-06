import re
import json

# File paths
input_file = 'output.txt'
output_file = 'timestamps.json'

# Step 1: Parse the File
with open(input_file, 'r') as file:
    content = file.read()

# Extract all Time Stamp values
time_stamps = re.findall(r"Time Stamp\s+:\s+([\d.]+)", content)

# Step 2: Count and Identify Unique Values
total_count = len(time_stamps)
unique_values = list(set(time_stamps))
unique_count = len(unique_values)

# Step 3: Prepare Data for JSON
data = {
    "total_count": total_count,
    "unique_count": unique_count,
    "unique_values": unique_values
}

# Save to JSON
with open(output_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Results saved to {output_file}")