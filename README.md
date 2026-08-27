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
</div>

A webapp for overlaying GPS/G-force telemetry HUDs onto action-cam footage: drag in a video,
trim the section you want, pick which telemetry widgets to draw (speedometer, lap timer, G-G
diagram, minimap, lap-vs-lap comparison), and download the result — with the HUD automatically
composited onto your real footage.

**Every one of those steps — trimming, joining split recordings, extracting telemetry,
detecting laps, drawing the HUD, compositing it onto your footage — runs entirely in your own
browser tab**, via WebCodecs, Canvas2D, and a couple of purpose-built media libraries (see
[Repo map](#repo-map) below). Your video is never uploaded anywhere; the backend that serves
this page has no video-processing code left in it at all.

## Run it

```sh
docker compose up --build
```

Then open http://localhost:7000.

Building locally takes well under a minute now — the backend is just a static-file server, with
no heavy Python dependencies to install. If you'd rather skip that anyway, every push to `main`
publishes a ready-to-run multi-arch image (see [CI/CD](#cicd) below) — pull it instead:

```sh
docker run -d --name routetracker -p 7000:7000 ghcr.io/jlleongarcia/py_routetracker:latest
```

### Accessing it from other devices over Tailscale

The container publishes port 7000 on all of the host's network interfaces (not just
`127.0.0.1`), so once it's running you can also reach it from any other device on your
Tailscale network at `http://<this-machine's-tailscale-ip-or-MagicDNS-name>:7000` — no
extra Docker network configuration needed. If it's not reachable, check that your host's
firewall (e.g. `ufw`) allows inbound connections on port 7000 from the `tailscale0`
interface.

> [!WARNING]
> Rendering depends on WebCodecs, which browsers only expose in a *secure context*
> (HTTPS, or the special-cased `localhost`/`127.0.0.1`). Plain `http://` access over
> Tailscale — including the MagicDNS URL above — is **not** a secure context, so
> rendering will fail there even though upload, trimming, and telemetry extraction work
> fine. Serve HTTPS instead, e.g. via [`tailscale serve`](https://tailscale.com/kb/1242/tailscale-serve)
> pointed at this container's port 7000 (gives you a `https://<magicdns-name>.<tailnet>.ts.net`
> URL with a real, trusted cert) or your own reverse proxy in front of the container.

### Install it as an app

PitLane is a Progressive Web App — open it in Chrome, Edge, or a Chromium-based mobile
browser and you'll get an install prompt (or use the browser's own "Install app" /
"Add to Home Screen" menu entry; iOS Safari only offers the latter, under the Share
sheet). Installed, it opens in its own window with no browser chrome, gets a real home
screen/dock icon, and — since every feature runs client-side — it keeps working fully
offline after the first load: no server round-trip is needed to upload, extract, render,
or download anything.

## Storage & privacy

Nothing is stored server-side, because nothing is ever sent server-side. Your video, GPX
file, and rendered output stay on your own device the entire time — the backend you're
talking to only ever serves the app's own JS/CSS/HTML.

## Supported telemetry sources

- **GoPro embedded GPS** — reads the camera's own embedded GPS/accelerometer metadata
  directly from the video file, in the browser (via `gpmf-extract` + `gopro-telemetry`),
  no separate file needed.
- **Video + separate GPX file** — for any other action cam (DJI, Insta360, older GoPros) or
  when you have a phone/Garmin/Polar GPX track instead. You can optionally give the video's
  real-world start time to sync the two automatically.

## Joining split recordings

Some action cams split one continuous recording into multiple files once it crosses a size
threshold (e.g. a chaptered GoPro clip: `GH010437.MP4`, `GH020437.MP4`, ...). Pick **"Join
split recording"** on the upload page, drop in all the parts, and they're combined into one
file — in the browser, via `mp4box.js` — before extraction runs. GoPro-numbered parts
(`GHccNNNN`/`GXccNNNN`) are auto-ordered by chapter, otherwise arrange them manually with the
up/down controls.

The join itself is a lossless, container-level concatenation: every track carries over
verbatim at its original sample description, including a GoPro's embedded GPMF telemetry
track that a naive re-encode join (or most video editors) would otherwise drop. Parts must
share the same tracks/codecs/resolution/frame rate (true for chapters of one recording);
mismatched files are rejected with a clear error before anything is joined.

## Repo map

```
backend/    Minimal FastAPI static-file server -- see backend/readme.md
frontend/   React + Vite + TypeScript webapp; frontend/src/lib/ is where everything runs:
              lib/telemetry/   GoPro GPMF + GPX parsing, resampling
              lib/laps/        lap detection, lap-vs-lap comparison
              lib/mp4/         video probing + the client-side lossless join
              lib/render/      the WebCodecs decode -> Canvas2D HUD draw -> encode -> mux
                                pipeline, run in workers/renderWorker.ts
```

See `backend/readme.md` for backend development/testing, and run `npm run dev` in
`frontend/` for frontend development (`npm run test` runs the Vitest suite covering the
telemetry/lap-detection ports).

## CI/CD

`.github/workflows/docker-multiarch.yml` builds and publishes a multi-architecture
(`linux/amd64` + `linux/arm64`) Docker image to GHCR — the arm64 build is what lets this run
on a Raspberry Pi or other ARM homelab box, not just x86 servers.

- **Triggers**: push to `main`, version tags (`v*.*.*`), pull requests, or manually
  (`workflow_dispatch`).
- **`test-backend`** and **`test-frontend`** run first — backend `pytest` (two tests, covering
  the static-file server's route-registration order) and the frontend's `npm run test`
  (Vitest) + `tsc` + `vite build`. Both must pass before anything is published.
- **`build-and-push`** only runs on an actual push/tag/manual dispatch — never on a pull
  request (a PR, especially from a fork, has no business publishing images, and forked PRs
  don't get write access to push to GHCR anyway). It publishes:
  - `ghcr.io/<owner>/py_routetracker:latest` — from the latest `main`.
  - `ghcr.io/<owner>/py_routetracker:<version>` — from a `v*.*.*` tag.
  - `ghcr.io/<owner>/py_routetracker:<short-sha>` — every build, for pinning to an exact
    commit.

The first time a workflow run publishes a package, GHCR makes it **private** by default —
go to the package's settings on GitHub and change its visibility to public if you want the
`docker pull` command above to work without authentication.

## Powered thanks to :heart:

[FastAPI](https://github.com/fastapi/fastapi) ·
[React](https://github.com/facebook/react) ·
[Leaflet](https://github.com/Leaflet/Leaflet) ·
[Recharts](https://github.com/recharts/recharts) ·
[mediabunny](https://github.com/Vanilagy/mediabunny) ·
[mp4box.js](https://github.com/gpac/mp4box.js) ·
[gpmf-extract](https://github.com/JuanIrache/gpmf-extract) ·
[gopro-telemetry](https://github.com/JuanIrache/gopro-telemetry) ·
[WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API)
