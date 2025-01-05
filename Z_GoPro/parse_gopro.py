import subprocess
import re

# Run exiftool to extract metadata
def run_exiftool(file_path):
    result = subprocess.run(['exiftool', '-ee', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8')

# Parse the extracted metadata to find relevant fields
def parse_metadata(metadata):
    gps_data = {
        'timestamps': [],
        'latitude': [],
        'longitude': [],
        'altitude': [],
        'speed': [],
        'speed_3d': []
    }
    
    # Regular expressions for matching relevant data
    timestamp_pattern = re.compile(r'Time Stamp\s*:\s*([0-9\.]+)')
    latitude_pattern = re.compile(r'GPS Latitude\s*:\s*([0-9]+ deg [0-9]+\' [0-9\.]+\" [NS])')
    longitude_pattern = re.compile(r'GPS Longitude\s*:\s*([0-9]+ deg [0-9]+\' [0-9\.]+\" [EW])')
    altitude_pattern = re.compile(r'GPS Altitude\s*:\s*([0-9\.]+ m)')
    speed_pattern = re.compile(r'GPS Speed\s*:\s*([0-9\.]+)')
    speed_3d_pattern = re.compile(r'GPS Speed 3D\s*:\s*([0-9\.]+)')
    
    # Extract and store matches
    gps_data['timestamps'] = timestamp_pattern.findall(metadata)
    gps_data['latitude'] = latitude_pattern.findall(metadata)
    gps_data['longitude'] = longitude_pattern.findall(metadata)
    gps_data['altitude'] = altitude_pattern.findall(metadata)
    gps_data['speed'] = speed_pattern.findall(metadata)
    gps_data['speed_3d'] = speed_3d_pattern.findall(metadata)
    
    return gps_data

# Save the parsed data to a text file
def save_to_file(gps_data, output_file):
    with open(output_file, 'w') as f:
        f.write("Timestamps:\n")
        for timestamp in gps_data['timestamps']:
            f.write(f"{timestamp}\n")
        
        f.write("\nGPS Latitude:\n")
        for lat in gps_data['latitude']:
            f.write(f"{lat}\n")
        
        f.write("\nGPS Longitude:\n")
        for lon in gps_data['longitude']:
            f.write(f"{lon}\n")
        
        f.write("\nGPS Altitude:\n")
        for alt in gps_data['altitude']:
            f.write(f"{alt}\n")
        
        f.write("\nGPS Speed:\n")
        for speed in gps_data['speed']:
            f.write(f"{speed}\n")
        
        f.write("\nGPS Speed 3D:\n")
        for speed_3d in gps_data['speed_3d']:
            f.write(f"{speed_3d}\n")

# Main function
def main():
    input_file = './home/jalcocert/Desktop/GX030390.MP4'  # Path to your video file
    output_file = 'gps_data.txt'   # Output text file to save the data
    
    # Run ExifTool and parse the metadata
    metadata = run_exiftool(input_file)
    gps_data = parse_metadata(metadata)
    
    # Save the parsed data to a file
    save_to_file(gps_data, output_file)
    print(f"GPS data saved to {output_file}")

# Run the script
if __name__ == "__main__":
    main()
