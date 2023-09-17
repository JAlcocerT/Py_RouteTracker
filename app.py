
import streamlit as st
import folium
from streamlit_folium import folium_static
import gpxpy
import pandas as pd
from io import BytesIO

# Create a Streamlit app
st.title("Py Route Tracker")

# Larger text with Markdown
st.write("## The world is waiting for your discovery")

# Create a left sidebar menu for file upload
st.sidebar.title("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload GPX files", type=["gpx"], accept_multiple_files=True)
hex_color = st.sidebar.color_picker("Select Route Color", "#00FF00")  # Default to green

# Initialize a global list to store coordinates
all_coordinates = []

# Subtitle after the first map
st.write("## A path is waiting to be discovered")

# Define a function to load and display GPX routes on the map
def PyLoadRoutes(gpx_file, hex_color):

    st.sidebar.write(f"Loading the GPX file: {gpx_file.name}")

    gpx_contents = gpx_file.read()
    gpx_file = BytesIO(gpx_contents)

    gpx = gpxpy.parse(gpx_file)

    route_info = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation
                })

    route_df = pd.DataFrame(route_info)

    new_coordinates = [tuple(x) for x in route_df[['latitude', 'longitude']].to_numpy()]

    # Append new coordinates to the global list of all coordinates
    all_coordinates.append(new_coordinates)

    # Clear the existing map
    m = folium.Map(
        location=[47, 32],  # Adjust the initial map center here
        zoom_start=5,
        tiles='OpenStreetMap',
        width=924,
        height=600
    )

    # Create separate polylines for each distinct sublist of coordinates
    for sublist in all_coordinates:
        folium.PolyLine(sublist, weight=3, color=hex_color).add_to(m)

    # Display the updated Folium map
    folium_static(m)

    st.sidebar.write("GPX file processed and displayed on the map")
    return(m)

# Process and display the uploaded GPX files
for uploaded_file in uploaded_files:
    if uploaded_file:
        PyLoadRoutes(uploaded_file, hex_color)  # You can customize the color here