
## Overlay

See the `comparison.md` file for a comparison of the different overlay methods.


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
python3.10 /home/jalcocert/Desktop/Py_RouteTracker/overlay/lap_timer_v4a.py #lap starts where lat lon of the given second of the video
```

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