
## Overlay

See the `comparison.md` file for a comparison of the different overlay methods.

## Circuit Mapper

The `circuit_mapper` scripts visualize the track geometry and export it for analysis.

> For the future **optimum path**!

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
