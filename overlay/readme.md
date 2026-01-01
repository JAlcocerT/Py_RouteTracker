
## Overlay

See the `comparison.md` file for a comparison of the different overlay methods.

This is based on **GPS speeds only**:

```sh
#python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/racing_hud_v3b.py
ffmpeg -i /home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020410.MP4 \
       -i /home/jalcocert/Desktop/Py_RouteTracker/overlay/racing_hud_v3b.mp4 \
       -filter_complex "[1:v]format=rgba,colorkey=0x000000:0.1:0.1[ckout];[0:v][ckout]overlay=W-w-50:H-h-50" \
       -codec:a copy \
       -preset superfast \
       racing_v3b_output.mp4

### Racing HUD v4 (Lap Integrated)
#This version includes the real-time **Lap Counter** and **Lap Timer**, plus it marks the laps on the speed graph.

python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/racing_hud_v4.py
ffmpeg -i /home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4 \
       -i /home/jalcocert/Desktop/Py_RouteTracker/overlay/racing_hud_v4.mp4 \
       -filter_complex "[1:v]format=rgba,colorkey=0x000000:0.1:0.1[ckout];[0:v][ckout]overlay=W-w-50:H-h-50" \
       -codec:a copy \
       -preset superfast \
       racing_v4_output.mp4

#python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v4a.py #lap starts where lat lon of the given second of the video
#python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v5.py #slices best lap

python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v6.py #possibility to compare 2 laps

#time python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/racing_hud_v6.py #now extracts everything (GPS and ACCL into a bin)
ffmpeg -i /home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX020410.MP4 \
       -i /home/jalcocert/Desktop/Py_RouteTracker/overlay/HUD_v6_GX020410.mp4 \
       -filter_complex "[1:v]format=rgba,colorkey=0x000000:0.1:0.1[ckout];[0:v][ckout]overlay=W-w-50:H-h-50" \
       -codec:a copy \
       -preset superfast \
       racing_v6_output_p1.mp4


ffmpeg -f concat -safe 0 \
  -i <(printf "file '$PWD/racing_v6_output_p1.mp4'\nfile '$PWD/racing_v6_output_p2.mp4'") \
  -c copy \
  racing_v6_combined.mp4
```


But it could be **based on the accelerations**:

Most overlay programs (like Telemetry Overlay or RaceRender) extract the ACCL (Accelerometer) stream from the GoPro's GPMD metadata.

This stream gives you Raw Acceleration in m/s² for:

* X (Left/Right): Cornering G-Force.
* Y (Up/Down): Vertical G-Force (bumps).
* Z (Forward/Back): Braking/Acceleration G-Force.

```sh
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/extract_gforce.py
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/gforce_hud_v1.py #this will create the gforce_hud_v1.mp4
ffmpeg -i /home/jalcocert/Desktop/Py_RouteTracker/Z_GoPro/GX030410.MP4 \
       -i /home/jalcocert/Desktop/Py_RouteTracker/overlay/GForce_HUD_v1.mp4 \
       -filter_complex "[1:v]format=rgba,colorkey=0x000000:0.1:0.1[ckout];[0:v][ckout]overlay=W-w-50:H-h-50" \
       -codec:a copy \
       -preset superfast \
       GForce_HUD_v1_combined.mp4
```

The ACCL (Accelerometer) and GYRO (Gyroscope) streams you just extracted are the exact raw sensor data the GoPro uses for its HyperSmooth Stabilization.

* ACCL (~200Hz): Detects linear forces (gravity, braking, bumps).
* GYRO (~400Hz+): Detects rotation (camera tilt/roll).

See one: https://youtu.be/3hX4JdDePfo



## Circuit Mapper

The `circuit_mapper` scripts visualize the track geometry and export it for analysis.

> For the future [**optimum path** project](https://github.com/JAlcocerT/optimum-path)!

Extracts a **Single Canonical Lap** (auto-detects start/finish loop) and calculates track boundaries (default 8m width).

**Outputs:**
1.  **`track_canonical_v3.png`**: Visual check of the canonical lap.
2.  **`track_canonical.csv`**: Human-readable text file.
    *   Columns: `center_x`, `center_y`, `left_x`, `left_y`, `right_x`, `right_y`.
    *   Coordinates are in meters relative to the track center.
3.  **`track_canonical.npz`**: Optimized binary for Python/NumPy.

**How to use `.npz` data:**

```python
import numpy as np

data = np.load('track_canonical.npz')
center = np.column_stack((data['center_x'], data['center_y']))
left_limit = np.column_stack((data['left_x'], data['left_y']))
right_limit = np.column_stack((data['right_x'], data['right_y']))
width = data['width']
```

Versions
*   `v1`: Basic shape plot + Leaflet HTML map.
*   `v2`: Exports full session data (multiple laps).
*   `v3`: Exports single clean lap (ideal for optimization).

```sh
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer.py #lap starts where video starts
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v4.py #lap starts where lat lon of the given second of the video
#python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v4a.py #adding max/min speeds of each lap

python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v5.py #extracts the telemtry txt on its own and you can use it to compare the best laps of a group of friends #https://youtu.be/Ae8CyefuxgY
#one after the other
ffmpeg -f concat -safe 0 \
  -i <(printf "file '$PWD/overlay/Best_Lap_4_81.33s.mp4'\nfile '$PWD/overlay/Best_Lap_1_78.61s_v5.mp4'") \
  -c copy overlay/Joined_Best_Laps.mp4

#one on top of the other (stacked)
ffmpeg -i overlay/Best_Lap_4_81.33s.mp4 \
       -i overlay/Best_Lap_1_78.61s_v5.mp4 \
       -filter_complex \
       "[0:v]tpad=stop_mode=clone:stop_duration=5[v0]; \
        [1:v]tpad=stop_mode=clone:stop_duration=5[v1]; \
        [v0][v1]vstack=inputs=2[v]" \
       -map "[v]" \
       -c:v libx264 -crf 23 -preset superfast \
       overlay/Stacked_Comparison_Fixed.mp4
```

That is the tolerance "bubble" size for detecting the Start/Finish line. LAP_DETECTION_RADIUS_M = 15.0 🎯

Since GPS data is rarely perfect (it drifts) and you might drive on the far left or right of the track, checking for an exact coordinate match almost never works.

Instead, the script asks: "Is the kart within 15 meters of the Start Point?"

15.0 meters: This radius is chosen to cover the entire track width (usually ~8-12m) plus a margin for GPS error.
If too small (e.g., 2m): The script might miss a lap if you take a wide line or the GPS drifts.
If too big (e.g., 50m): It might accidentally count a lap if the track loops back close to the start line (e.g., a hairpin turn near the pits).
15m is usually the "sweet spot" for ensuring every lap is counted correctly!


```sh
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/prepend_intro.py
```

### Manual FFmpeg Commands (No Script)

If you prefer running it yourself, here is the magic 2-step process (avoids re-encoding the main video):

**1. Generate the Intro (Must match GoPro Codec)**

```bash
ffmpeg -y -loop 1 -i overlay/lap_analysis_v4a.png \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v libx265 -tag:v hvc1 -pix_fmt yuvj420p \
  -vf "scale=3840:2160,setsar=1" \
  -r 60000/1001 \
  -c:a aac -ar 48000 -ac 2 \
  -t 2 \
  -x265-params log-level=error:crf=20 \
  intro_temp.mp4
```

**2. Concatenate (One-Liner, No Text File)**

You can feed the file list directly into FFmpeg (Bash only):

```bash
ffmpeg -f concat -safe 0 -i <(printf "file '$PWD/intro_temp.mp4'\nfile '$PWD/Z_GoPro/GX030410.MP4'") -c copy full_video.mp4
```