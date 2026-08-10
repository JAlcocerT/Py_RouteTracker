import gpxpy
import gpxpy.gpx
import webbrowser
import os

GPX_FILE = 'Data_Komobi/komobi_sample_data.gpx'
LEAFLET_HTML = 'gpx_route_map.html'

LEAFLET_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>GPX Route Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
    <style>
        #map { height: 90vh; width: 100vw; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{{center_lat}}, {{center_lon}}], 15);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        var latlngs = {{latlngs}};
        var polyline = L.polyline(latlngs, {color: 'blue'}).addTo(map);
        map.fitBounds(polyline.getBounds());
    </script>
</body>
</html>
'''

def extract_track_points(gpx):
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append([pt.latitude, pt.longitude])
    return points

def generate_leaflet_map(points, output_file):
    if not points:
        print("No track points found.")
        return
    center_lat = points[0][0]
    center_lon = points[0][1]
    latlngs = str(points)
    html = LEAFLET_TEMPLATE.replace('{{center_lat}}', str(center_lat)) \
                           .replace('{{center_lon}}', str(center_lon)) \
                           .replace('{{latlngs}}', latlngs)
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Map saved to {output_file}")
    webbrowser.open('file://' + os.path.abspath(output_file))

def main():
    with open(GPX_FILE, 'r') as f:
        gpx = gpxpy.parse(f)
    points = extract_track_points(gpx)
    generate_leaflet_map(points, LEAFLET_HTML)

if __name__ == '__main__':
    main()
