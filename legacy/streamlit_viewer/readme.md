# Archived: Streamlit GPX viewer

The original tool this repo shipped: a basic Streamlit app for uploading GPX files and
viewing routes/elevation on a map — no video, no telemetry overlay. Superseded by the webapp
(`../../backend/` + `../../frontend/`), which covers this (map preview during upload) plus
the actual video-overlay pipeline.

- `app.py` — the Streamlit app.
- `Dockerfile`, `requirements.txt` — its deployment setup (the CI workflows that built and
  pushed this image have been removed, since they targeted the root `Dockerfile` which is now
  the webapp's).
- `Z_Deploy_me/` — a docker-compose file for running it.
- `original_root_readme.md` — this repo's README before the webapp rewrite, kept for history.
