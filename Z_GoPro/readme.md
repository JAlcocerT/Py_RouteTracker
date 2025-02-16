Extracting **gopro metadata**.

> Extended [blog post](https://jalcocert.github.io/JAlcocerT/dji-oa5pro-firmware-updates/#extracting-telemetry-data-from-gph9)

* https://medium.com/@jrballesteros/a-simple-guide-to-extract-gps-information-from-gopro-photos-and-videos-cf6edf6dc601
    * https://github.com/exiftool/exiftool/blob/master/fmt_files/gpx.fmt

```sh
sudo apt-get install libimage-exiftool-perl #install exif
#exiftool -ee ./GX030390.MP4 #you will see it on CLI
exiftool -ee ./GX030390.MP4 > output-GX030390.txt #saves it
```

---

## Venv Setup


```sh
#python -m venv gopro_venv #create the venv
python3 -m venv gopro_venv #create the venv

#gopro_venv\Scripts\activate #activate venv (windows)
source gopro_venv/bin/activate #(linux)
```

Install them with:

```sh
#pip install beautifulsoup4 openpyxl pandas numpy==2.0.0
pip install -r requirements.txt #all at once
#pip freeze | grep langchain

#pip show beautifulsoup4
pip list
#pip freeze > requirements.txt #generate a txt with the ones you have!
```

```sh
source .env

#export OPENAI_API_KEY="your-api-key-here"
#set OPENAI_API_KEY=your-api-key-here
#$env:OPENAI_API_KEY="your-api-key-here"
echo $OPENAI_API_KEY
```

---

Sample Video:

[![Recording the Data](https://img.youtube.com/vi/Ku3y3NJJURw/0.jpg)](https://www.youtube.com/watch?v=Ku3y3NJJURw)