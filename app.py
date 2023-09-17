import streamlit as st
import folium
from streamlit_folium import folium_static
import gpxpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly graph objects
from io import BytesIO

# Create a Streamlit app
st.title("Py Route Tracker")

# Larger text with Markdown
st.write("## The world is waiting for your discovery!")

# Create a left sidebar menu for file upload
st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload GPX files", type=["gpx"], accept_multiple_files=True)
hex_color = st.sidebar.color_picker("Select Route Color", "#35BD07")  # Default to green

# Initialize a global list to store coordinates and their respective file names
all_coordinates = []
file_names = []

# Create a Folium map
m = folium.Map(
    location=[47, 32],  # Adjust the initial map center here
    zoom_start=1,
    tiles='cartodb positron',
    width=924,
    height=600
)

# Display the Folium map using streamlit-folium
folium_static(m)

# Subtitle after the first map
st.write("## These paths are waiting to be discovered:")

# Define a function to load and display GPX routes on the map
def PyLoadRoutes(uploaded_file, hex_color):

    st.sidebar.write(f"Loading the GPX file: {uploaded_file.name}")

    gpx_contents = uploaded_file.read()
    gpx_file = BytesIO(gpx_contents)

    gpx = gpxpy.parse(gpx_file)

    route_info = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation if hasattr(point, 'elevation') else None
                })

    route_df = pd.DataFrame(route_info)

    new_coordinates = [tuple(x) for x in route_df[['latitude', 'longitude', 'elevation']].to_numpy()]

    # Append new coordinates to the global list of all coordinates
    all_coordinates.append(new_coordinates)

    # Append the GPX file name to the list of file names
    file_names.append(uploaded_file.name)

    # Clear the existing map
    m = folium.Map(
        location=[47, 32],  # Adjust the initial map center here
        zoom_start=5,
        tiles='OpenStreetMap',
        width=924,
        height=600
    )

    # Create separate polylines for each distinct sublist of coordinates
    for sublist, file_name in zip(all_coordinates, file_names):
        folium.PolyLine([coord[:2] for coord in sublist], weight=3, color=hex_color, tooltip=file_name).add_to(m)

    # Display the updated Folium map
    folium_static(m)

    st.sidebar.write("GPX file processed and displayed on the map")

    # Plot elevation vs. distance
    plot_elevation_vs_distance(all_coordinates, file_names)

# Define a function to plot elevation vs. distance using Plotly
def plot_elevation_vs_distance(coordinates_list, file_names):
    fig = go.Figure()  # Create a Plotly figure
    
    for i, (coordinates, file_name) in enumerate(zip(coordinates_list, file_names)):
        distances = []
        elevations = []
        total_distance = 0

        for j in range(len(coordinates) - 1):
            lat1, lon1, elevation1 = coordinates[j]
            lat2, lon2, elevation2 = coordinates[j + 1]

            # Calculate distance between two points (Haversine formula)
            d = haversine(lat1, lon1, lat2, lon2)

            total_distance += d
            distances.append(total_distance)
            elevations.append(elevation1)  # Use elevation1

        fig.add_trace(go.Scatter(x=distances, y=elevations, mode='lines', name=f'Route {file_name}'))

    fig.update_layout(
        xaxis_title="Distance (meters)",
        yaxis_title="Elevation (meters)",
        title="Elevation vs. Distance"
    )

    # Display the Plotly figure
    st.plotly_chart(fig)

# Haversine formula to calculate distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    import math

    # Radius of the Earth in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Process and display the uploaded GPX files
for uploaded_file in uploaded_files:
    if uploaded_file:
        PyLoadRoutes(uploaded_file, hex_color)  # You can customize the color here