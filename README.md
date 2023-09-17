# Py_RouteTracker

Analyzing location and routes data with Python.
## Repository Structure

* Main Folder:
    * app.py - A streamlit app to interactively load and display your .GPX routes interactively
    * Py_Route_to_HTML.ipynb - Visualize your GPX file data in **OpenStreetMap** with **Folium**, also export it.
    * Py_RouteTracker.ipynb 
    * Py_RoutePolar.ipynb

* ./EDA:
    * My learning process
* ./Data ./Data_My_Routes and ./Data_Polar
    * Sample GPX files
    * Sample Polar Data
    * Sample GEOJSON files: in /HU-GR_geojson folder
* ./Data_Maps:
    * Sample shp files: in /NUTS_RG_* folder

## Try Me: it is F/OSS!

* Deploy me using Docker: 
    * [Why Docker?](https://fossengineer.com/docker-first-steps-guide-for-data-analytics/)
    * This Repository has automatic CI/CD with Github Actions
        * The magic happens at /.github/workflows/ci-cd.yml
        * Learn [how to setup Github CI/CD](https://fossengineer.com/docker-github-actions-cicd/#github-workflows)
* You can use any of the notebooks, like Py_Route_to_HTML, directly from Google Colaboratory:

<a 
 href="https://colab.research.google.com/github/JAlcocerT/Py_RouteTracker/blob/main/Py_Route_to_HTML.ipynb"
 target="_parent">
<img 
 src="https://colab.research.google.com/assets/colab-badge.svg"
alt="Open In Colab"/>
</a>

* Analyze your sport performance using the Py_Route_Polar.ipynb
    * Wrote about this in my Tech Blog: <https://fossengineer.com/polar-data-python-analysis/>

## Powered Thanks To :heart:

* [Plotly](https://github.com/plotly/plotly.py)
* [Folium](https://github.com/python-visualization/folium)
* [GPXpy](https://github.com/tkrajina/gpxpy/tree/dev)
* [Streamlit](https://github.com/streamlit/streamlit)

### Recommended Resources

* <https://www.openstreetmap.org/>
* <https://github.com/thedirtyfew/dash-leaflet>

## :loudspeaker: Ways to Contribute 

* Please feel free to fork the code - try it out for yourself and improve or add others tabs. The data that is queried give many possibilities to create awsome interactive visualizations.

* Support the Projects that made possible this App, thanks to their fantastic job, this have been possible.

* Support extra evening code sessions:

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/FossEngineer)


## :scroll: License

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License (GPL) version 3.0:

    Freedom to use: You can use the software for any purpose, without any restrictions.
    Freedom to study and modify: You can examine the source code, learn from it, and modify it to suit your needs.
    Freedom to share: You can share the original software or your modified versions with others, so they can benefit from it too.
    Copyleft: When you distribute the software or any derivative works, you must do so under the same GPL-3.0 license. This ensures that the software and its derivatives remain free and open-source.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.