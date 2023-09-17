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