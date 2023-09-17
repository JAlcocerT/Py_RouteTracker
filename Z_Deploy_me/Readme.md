

docker image build -t fossengineer/py_routetracker:v1.0 .
docker image build -t fossengineer/py_routetracker:v1.0 -f ./Z_Deploy_me/Dockerfile .



```yml
version: '3.8'

services:
  my_python_dev_container:
    image: fossengineer/py_routetracker:v1.0
    container_name: routetracker
    ports:
      - "8509:8501"
    working_dir: /app
    command: streamlit run app.py
    #command: python3 app.py
    #command: tail -f /dev/null #keep it running
```