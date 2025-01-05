<div align="center">
  <h1>Py_RouteTracker</h1>
</div>

<div align="center">
  <h3>Analyzing location and routes data with Python.</h3>
</div>

<div align="center">
  <a href="https://github.com/JAlcocerT/Py_RouteTracker?tab=GPL-3.0-1-ov-file" style="margin-right: 5px;">
    <img alt="Code License" src="https://img.shields.io/badge/License-GPLv3-blue.svg" />
  </a>
  <a href="https://github.com/JAlcocerT/Py_RouteTracker/actions/workflows/ci-cd.yml" style="margin-right: 5px;">
    <img alt="GH Actions Workflow" src="https://github.com/JAlcocerT/Py_RouteTracker/actions/workflows/ci-cd.yml/badge.svg" />
  </a>
  <a href="https://GitHub.com/JAlcocerT/Py_RouteTracker/graphs/commit-activity" style="margin-right: 5px;">
    <img alt="Mantained" src="https://img.shields.io/badge/Maintained%3F-no-grey.svg" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3819/">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.10-blue.svg" />
  </a>
</div>


## Repository Structure


<details>
  <summary>Main Folder</summary>
  &nbsp;

* `app.py` - A streamlit app to interactively load and display your .GPX routes interactively
* `Py_Route_to_HTML.ipynb` - Visualize your GPX file data in **OpenStreetMap** with **Folium**, also export it.
* `Py_RouteTracker.ipynb` 
* `Py_RoutePolar.ipynb` - Analyze Polar Data with Python

</details>

* `./EDA`:
    * My learning process
* `./Data` .`/Data_My_Routes` `./Data_Polar` `./Data_Kart`
    * Sample GPX files
    * Sample Polar Data
    * Sample GEOJSON files: in `/HU-GR_geojson` folder
* `./Data_PhyPhox`
    * Sample csv Data export from **PhyPhox** (Physical Phone Experiments)
* `./Data_Maps`:
    * Sample shp files: in `/NUTS_RG_*`folder

## Try Me: it is F/OSS!

* Deploy me using Docker: 
    * [Why Docker?](https://fossengineer.com/docker-first-steps-guide-for-data-analytics/)
    * This Repository has automatic CI/CD with Github Actions
        * The magic happens at `./github/workflows/ci-cd.yml`
        * Learn [how to setup Github CI/CD](https://jalcocert.github.io/JAlcocerT/github-actions-use-cases)
* You can use any of the notebooks, like Py_Route_to_HTML, directly from Google Colaboratory:

<div style="text-align: center;">
  <a href="https://colab.research.google.com/github/JAlcocerT/Py_RouteTracker/blob/main/Py_Route_to_HTML.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</div>

* Analyze your sport performance using the `Py_Route_Polar.ipynb`
    * Wrote about this in my Tech Blog: <https://jalcocert.github.io/JAlcocerT/polar-data-python-analysis//>

## Powered Thanks To :heart:

* [Plotly](https://github.com/plotly/plotly.py)
* [Folium](https://github.com/python-visualization/folium)
* [GPXpy](https://github.com/tkrajina/gpxpy/tree/dev)
* [Streamlit](https://github.com/streamlit/streamlit)
* [PhyPhox](https://github.com/phyphox/phyphox-android)

### Recommended Resources

These 2 projects are fantastic to work with:

* <https://www.openstreetmap.org/>
* <https://github.com/thedirtyfew/dash-leaflet>

## :loudspeaker: Ways to Contribute 

* Please feel free to fork the code - try it out for yourself and improve or add others tabs. The data that is queried give many possibilities to create awsome interactive visualizations.

* Support the Projects that made possible this App, thanks to their fantastic job, this have been possible.

* Support extra evening code sessions:

<div align="center">
  <a href="https://ko-fi.com/Z8Z1QPGUM">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi">
  </a>
</div>


## :scroll: License

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License (GPL) version 3.0:

    Freedom to use: You can use the software for any purpose, without any restrictions.
    Freedom to study and modify: You can examine the source code, learn from it, and modify it to suit your needs.
    Freedom to share: You can share the original software or your modified versions with others, so they can benefit from it too.
    Copyleft: When you distribute the software or any derivative works, you must do so under the same GPL-3.0 license. This ensures that the software and its derivatives remain free and open-source.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.