
## Overlay

See the `comparison.md` file for a comparison of the different overlay methods.

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
```

That is the tolerance "bubble" size for detecting the Start/Finish line. LAP_DETECTION_RADIUS_M = 15.0 🎯

Since GPS data is rarely perfect (it drifts) and you might drive on the far left or right of the track, checking for an exact coordinate match almost never works.

Instead, the script asks: "Is the kart within 15 meters of the Start Point?"

15.0 meters: This radius is chosen to cover the entire track width (usually ~8-12m) plus a margin for GPS error.
If too small (e.g., 2m): The script might miss a lap if you take a wide line or the GPS drifts.
If too big (e.g., 50m): It might accidentally count a lap if the track loops back close to the start line (e.g., a hairpin turn near the pits).
15m is usually the "sweet spot" for ensuring every lap is counted correctly!