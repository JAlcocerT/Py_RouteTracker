# GPX Route Visualizer

This folder contains a sample GPX file (`komobi_sample_data.gpx`) and a Python script to visualize its route on an interactive map using Leaflet.js.

## Features
- Parses a GPX file and extracts route/track points
- Generates an HTML map with the route drawn as a polyline
- Uses Leaflet.js and OpenStreetMap tiles
- Automatically opens the map in your web browser

## Requirements
- Python 3.x
- [gpxpy](https://pypi.org/project/gpxpy/)

Install the required Python package (if you haven't already):

```bash
#pip install gpxpy
uv init uv add gpxpy
uv sync
uv run sample_gpx_reader.py
```

## Usage

1. Make sure `sample_gpx_reader.py` and `komobi_sample_data.gpx` are in this folder.
2. Run the script:
   ```bash
   python3 sample_gpx_reader.py
   ```
3. An HTML file (`gpx_route_map.html`) will be generated and opened in your default browser, showing the route from the GPX file.

## Customization

- To use your own GPX file, replace `komobi_sample_data.gpx` with your file and update the filename in the script if needed.
- The script can be extended to show waypoints, change map styles, or export data in other formats.