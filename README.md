<div align="center">
  <h1>Py_RouteTracker</h1>
  <h3>Telemetry Overlay Studio — drag in action-cam footage, overlay GPS/G-force HUDs, download.</h3>
</div>

<div align="center">
  <a href="https://github.com/jlleongarcia/Py_RouteTracker/actions/workflows/docker-multiarch.yml">
    <img alt="CI/CD" src="https://github.com/jlleongarcia/Py_RouteTracker/actions/workflows/docker-multiarch.yml/badge.svg" />
  </a>
  <a href="https://github.com/jlleongarcia/Py_RouteTracker/pkgs/container/py_routetracker">
    <img alt="Container" src="https://img.shields.io/badge/ghcr.io-py__routetracker-blue?logo=docker" />
  </a>
  <a href="https://github.com/jlleongarcia/Py_RouteTracker?tab=AGPL-3.0-1-ov-file">
    <img alt="Code License" src="https://img.shields.io/badge/License-AGPLv3-blue.svg" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.12-blue.svg" />
  </a>
</div>

A local, self-hosted webapp for overlaying GPS/G-force telemetry HUDs onto action-cam
footage: drag in a video, trim the section you want, pick which telemetry widgets to draw
(speedometer, lap timer, G-G diagram, minimap, session graph, lap-vs-lap comparison), and
download the result — with the HUD automatically composited onto your real footage.

## Run it

```sh
docker compose up --build
```

Then open http://localhost:7000.

Building locally takes a few minutes (it compiles the frontend and installs the backend's
Python deps, including matplotlib). If you'd rather skip that, every push to `main` publishes
a ready-to-run multi-arch image (see [CI/CD](#cicd) below) — pull it instead:

```sh
docker run -d --name routetracker -p 7000:7000 -v routetracker_data:/data \
  ghcr.io/jlleongarcia/py_routetracker:latest
```

### Accessing it from other devices over Tailscale

The container publishes port 7000 on all of the host's network interfaces (not just
`127.0.0.1`), so once it's running you can also reach it from any other device on your
Tailscale network at `http://<this-machine's-tailscale-ip-or-MagicDNS-name>:7000` — no
extra Docker network configuration needed. If it's not reachable, check that your host's
firewall (e.g. `ufw`) allows inbound connections on port 7000 from the `tailscale0`
interface.

## Storage & privacy

This isn't video hosting — uploaded source videos and rendered output are **temporary**.
They're kept just long enough to render and download (default **60 minutes** from upload,
regardless of whether you've downloaded yet), then automatically deleted by a background
sweeper; nothing is kept in permanent storage on the host. Per-render scratch files (the
trimmed clip and the intermediate HUD frame images) are deleted immediately once a render
finishes, before that window even starts. The download page tells you exactly when your
file will be removed.

Tune this with environment variables on the `routetracker` service in `docker-compose.yml`:
- `ROUTETRACKER_RETENTION_MINUTES` (default `60`) — how long a video/render is kept.
- `ROUTETRACKER_SWEEP_INTERVAL_SECONDS` (default `300`) — how often the cleanup pass runs.

## Supported telemetry sources

- **GoPro embedded GPS** — reads the camera's own embedded GPS/accelerometer metadata
  (via `exiftool`/`ffmpeg`, no separate file needed).
- **Video + separate GPX file** — for any other action cam (DJI, Insta360, older GoPros) or
  when you have a phone/Garmin/Polar GPX track instead. You can optionally give the video's
  real-world start time to sync the two automatically.

## Repo map

```
backend/    FastAPI service: telemetry extraction, lap detection, HUD rendering, job queue
frontend/   React + Vite + TypeScript webapp
legacy/     Archived prototype scripts (overlay/) and the original Streamlit GPX viewer
research/   Sample data and exploratory notebooks this project grew out of; not maintained
```

`backend/` and `frontend/` are the project — see `backend/readme.md` for backend
development/testing, and run `npm run dev` in `frontend/` for frontend development.
Everything under `legacy/` and `research/` is historical context, kept for reference but not
part of the running app.

## CI/CD

`.github/workflows/docker-multiarch.yml` builds and publishes a multi-architecture
(`linux/amd64` + `linux/arm64`) Docker image to GHCR — the arm64 build is what lets this run
on a Raspberry Pi or other ARM homelab box, not just x86 servers.

- **Triggers**: push to `main`, version tags (`v*.*.*`), pull requests, or manually
  (`workflow_dispatch`).
- **`test-backend`** and **`test-frontend`** run first — real backend `pytest` (with
  `ffmpeg`/`exiftool` installed on the runner, so the full extract → render → composite
  pipeline is actually exercised, not just the parts that don't need those binaries) and a
  frontend `tsc` + `vite build`. Both must pass before anything is published.
- **`build-and-push`** only runs on an actual push/tag/manual dispatch — never on a pull
  request (a PR, especially from a fork, has no business publishing images, and forked PRs
  don't get write access to push to GHCR anyway). It publishes:
  - `ghcr.io/<owner>/py_routetracker:latest` — from the latest `main`.
  - `ghcr.io/<owner>/py_routetracker:<version>` — from a `v*.*.*` tag.
  - `ghcr.io/<owner>/py_routetracker:<short-sha>` — every build, for pinning to an exact
    commit.

This replaces the repo's original three CI/CD workflows (`CICD-GHA_MultiArch.yml`, `ci-cd.yml`,
`github-runner.yml`, archived along with the Streamlit app they built — see
`legacy/streamlit_viewer/`), which were near-duplicates of each other, published to a
hardcoded `jalcocert/...` GHCR namespace regardless of who forked the repo, required manually
creating and rotating a personal access token (`secrets.CICD_PyRouteTracker_Token`), and never
ran a test suite before publishing. This one uses the automatically-provided `GITHUB_TOKEN`
(no setup needed beyond forking — GitHub Actions has push access to your own fork's GHCR by
default) and targets whichever repo it actually runs in.

The first time a workflow run publishes a package, GHCR makes it **private** by default —
go to the package's settings on GitHub and change its visibility to public if you want the
`docker pull` command above to work without authentication.

## Powered thanks to :heart:

[FastAPI](https://github.com/fastapi/fastapi) ·
[React](https://github.com/facebook/react) ·
[Leaflet](https://github.com/Leaflet/Leaflet) ·
[Recharts](https://github.com/recharts/recharts) ·
[matplotlib](https://github.com/matplotlib/matplotlib) ·
[gpxpy](https://github.com/tkrajina/gpxpy) ·
[ffmpeg](https://ffmpeg.org/) · [ExifTool](https://exiftool.org/)
