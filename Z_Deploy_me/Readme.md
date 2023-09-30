# Deploy Me!

## [Build](https://fossengineer.com/building-docker-container-images/) Me

* Get Docker installed
* Clone this repository
* Execute:

```sh
#docker image build -t fossengineer/py_routetracker:v1.0 .
docker image build -t py_routetracker:v1.0 -f ./Z_Deploy_me/Dockerfile .
```


* And then: docker run -p 8501:8501 py_routetracker:v1.0
* Go to your browser and have a look to: localhost:8501

## Use my Github Container Registry Image

* I have used **GH Actions** to generate the package: <https://github.com/JAlcocerT/Py_RouteTracker/pkgs/container/py_routetracker>
  * If you are doing this for the first time, remember to connect the package with your repository (and to make it public, if you want).


```yml
version: '3.8'

services:
  route_tracker_streamlit_app:
    image:  ghcr.io/jalcocert/py_routetracker:v1.0 #fossengineer/py_routetracker:v1.0
    container_name: py_routetracker
    ports:
      - "8509:8501"
    working_dir: /app
    command: streamlit run app.py
    #command: python3 app.py
    #command: tail -f /dev/null #keep it running
```