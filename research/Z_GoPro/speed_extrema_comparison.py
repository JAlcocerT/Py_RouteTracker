import numpy as np
import matplotlib.pyplot as plt

def find_local_extrema(speeds, window=5, mode='max'):
    """
    Find local maxima or minima in a list of speeds.
    
    Args:
        speeds: List of speed values
        window: Number of points to consider on each side
        mode: 'max' to find local maxima, 'min' for local minima
        
    Returns:
        List of (index, speed) tuples for the extrema points
    """
    extrema = []
    for i in range(window, len(speeds)-window):
        center = speeds[i]
        left = speeds[i-window:i]
        right = speeds[i+1:i+window+1]
        
        if mode == 'max' and all(center >= x for x in left + right):
            extrema.append((i, center))
        elif mode == 'min' and all(center <= x for x in left + right):
            extrema.append((i, center))
    return extrema

def plot_speed_extrema_comparison(file1, file2, label1="File 1", label2="File 2", 
                                 time_interval=1, time_range=None, window=5):
    """
    Compare local maxima and minima between two speed profiles.
    
    Args:
        file1, file2: Paths to the GPS data files
        label1, label2: Labels for the legend
        time_interval: Time between measurements in seconds
        time_range: Tuple of (start_time, end_time) in seconds to filter the data
        window: Number of points to consider on each side when finding extrema
    """
    # Extract and filter speeds
    speeds1 = extract_gps_speeds(file1)
    speeds2 = extract_gps_speeds(file2)
    
    # Create time arrays
    time1 = np.arange(0, len(speeds1)) * time_interval
    time2 = np.arange(0, len(speeds2)) * time_interval
    
    # Apply time filtering if requested
    if time_range:
        start_idx1 = int(time_range[0] / time_interval)
        end_idx1 = min(int(time_range[1] / time_interval), len(speeds1))
        start_idx2 = int(time_range[0] / time_interval)
        end_idx2 = min(int(time_range[1] / time_interval), len(speeds2))
        
        speeds1 = speeds1[start_idx1:end_idx1]
        time1 = time1[start_idx1:end_idx1]
        speeds2 = speeds2[start_idx2:end_idx2]
        time2 = time2[start_idx2:end_idx2]
    
    # Find local maxima and minima
    max1 = find_local_extrema(speeds1, window, 'max')
    min1 = find_local_extrema(speeds1, window, 'min')
    max2 = find_local_extrema(speeds2, window, 'max')
    min2 = find_local_extrema(speeds2, window, 'min')
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot both speed profiles
    plt.plot(time1, speeds1, 'b-', alpha=0.5, label=f'{label1} Speed')
    plt.plot(time2, speeds2, 'r-', alpha=0.5, label=f'{label2} Speed')
    
    # Plot extrema
    if max1:
        max1_times = [time1[i] for i, _ in max1]
        max1_speeds = [s for _, s in max1]
        plt.scatter(max1_times, max1_speeds, c='blue', marker='^', s=100, 
                   label=f'{label1} Local Max (Window={window})')
    
    if min1:
        min1_times = [time1[i] for i, _ in min1]
        min1_speeds = [s for _, s in min1]
        plt.scatter(min1_times, min1_speeds, c='blue', marker='v', s=100, 
                   label=f'{label1} Local Min (Window={window})')
    
    if max2:
        max2_times = [time2[i] for i, _ in max2]
        max2_speeds = [s for _, s in max2]
        plt.scatter(max2_times, max2_speeds, c='red', marker='^', s=100, 
                   label=f'{label2} Local Max (Window={window})')
    
    if min2:
        min2_times = [time2[i] for i, _ in min2]
        min2_speeds = [s for _, s in min2]
        plt.scatter(min2_times, min2_speeds, c='red', marker='v', s=100, 
                   label=f'{label2} Local Min (Window={window})')
    
    # Add statistics table
    stats_data = [
        ['Local Max Count', len(max1), len(max2)],
        ['Local Min Count', len(min1), len(min2)],
        ['Avg Max Speed', 
         f"{np.mean([s for _, s in max1]):.1f} km/h" if max1 else 'N/A', 
         f"{np.mean([s for _, s in max2]):.1f} km/h" if max2 else 'N/A'],
        ['Avg Min Speed', 
         f"{np.mean([s for _, s in min1]):.1f} km/h" if min1 else 'N/A', 
         f"{np.mean([s for _, s in min2]):.1f} km/h" if min2 else 'N/A']
    ]
    
    table = plt.table(cellText=stats_data,
                     colLabels=['Statistic', label1, label2],
                     loc='bottom',
                     bbox=[0, -0.3, 1, 0.25])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add labels and title
    time_range_text = f" (Time: {time_range[0]}-{time_range[1]}s)" if time_range else ""
    plt.title(f'Speed Extrema Comparison{time_range_text}\nWindow Size: {window} points')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speed (km/h)')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print(f"EXTREMA COMPARISON (Window Size: {window} points)")
    print("="*60)
    print(f"{label1}: {len(max1)} local maxima, {len(min1)} local minima")
    if max1:
        print(f"  - Max speeds: {', '.join([f'{s:.1f} km/h' for _, s in max1])}")
    if min1:
        print(f"  - Min speeds: {', '.join([f'{s:.1f} km/h' for _, s in min1])}")
    
    print(f"\n{label2}: {len(max2)} local maxima, {len(min2)} local minima")
    if max2:
        print(f"  - Max speeds: {', '.join([f'{s:.1f} km/h' for _, s in max2])}")
    if min2:
        print(f"  - Min speeds: {', '.join([f'{s:.1f} km/h' for _, s in min2])}")

def extract_gps_speeds(file_path):
    """
    Extract GPS speed data from exiftool output file.
    
    Args:
        file_path: Path to the exiftool output file
        
    Returns:
        List of speed values in km/h
    """
    import re
    try:
        with open(file_path, 'r', encoding='utf-16-le') as file:
            content = file.read()
        speed_pattern = r"GPS Speed\s+:\s+([\d.]+)"
        speeds = re.findall(speed_pattern, content)
        return [float(speed) for speed in speeds]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    file1 = "output-kart25dec-1b.txt"
    file2 = "output-kart25dec-2b.txt"
    
    # Compare with default window size (5 points on each side)
    plot_speed_extrema_comparison(
        file1, file2, 
        label1="Run 1", 
        label2="Run 2",
        time_interval=1,
        time_range=(3700, 4500),  # Example time range
        window=90  # Number of points to consider on each side
    )
