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

### Accessing it from other devices

The container publishes port 7000 on all of the host's network interfaces (not just
`127.0.0.1`), so it's reachable from other machines on your LAN or Tailscale network
straight away — no extra Docker network configuration needed. If it isn't, check that the
host firewall (e.g. `ufw`) allows inbound connections on port 7000.

> [!IMPORTANT]
> **Reaching it over plain `http://` at an IP or hostname will break rendering.** You'll
> get: *"This page can't decode or encode video because it's loaded over an insecure
> connection."*
>
> This isn't something the app can work around. Rendering needs
> [WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API), and browsers
> only expose it in a **secure context** — HTTPS, or the specially-allowed `localhost` /
> `127.0.0.1`. A private LAN or Tailscale IP over `http://` is *not* a secure context: the
> browser doesn't care that the address is unroutable from the internet, only that the
> origin isn't authenticated. Upload, trimming and telemetry extraction still work; only
> rendering fails. Installing it as a PWA won't work either, for the same reason.
>
> On the host itself, `http://localhost:7000` works with no setup at all. For anything
> else, serve it over HTTPS — two easy routes below.

#### Option A: `tailscale serve` (private, recommended)

Keeps the app reachable only by your own devices, and issues a real, trusted certificate
automatically. First enable **HTTPS Certificates** for your tailnet (admin console → DNS),
then on the host:

```sh
sudo tailscale serve --bg --https=8443 7000
```

That publishes it at `https://<machine>.<your-tailnet>.ts.net:8443/`. Undo with
`sudo tailscale serve --https=8443 off`.

Use a **port** (`--https=8443`) rather than a sub-path (`--set-path`): the app's assets are
served from absolute paths and its PWA manifest declares `scope: "/"`, so mounting it under
a sub-path breaks asset resolution and the service worker's scope. If port 443 isn't already
taken by another `serve` mapping, plain `--bg 7000` works too and drops the `:8443`.

This is `serve`, not `funnel` — nothing is published to the public internet.

#### Option B: Cloudflare Tunnel (reachable anywhere)

Also gives a valid certificate and a working secure context, and unlike Tailscale it works
from devices outside your network. The trade-off is that it *is* internet-facing, so put
[Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) in
front of it unless you genuinely want it open. For a throwaway test,
`cloudflared tunnel --url http://localhost:7000` prints an ephemeral HTTPS URL needing no
domain.

Two settings will otherwise bite you:

- **Turn Rocket Loader off.** It rewrites and defers scripts, which breaks the ES module
  worker the renderer runs in.
- **Bypass cache for `/sw.js`.** Cloudflare caches `.js` by extension, and a cached service
  worker will pin visitors to a stale build indefinitely.

Your footage is unaffected either way: every feature runs client-side, so whichever route
you pick only ever serves the app's own JS/CSS/HTML/wasm. No video or GPX data is uploaded
(see [Storage & privacy](#storage--privacy)).

### Install it as an app

PitLane is a Progressive Web App — open it in Chrome, Edge, or a Chromium-based mobile
browser and you'll get an install prompt (or use the browser's own "Install app" /
"Add to Home Screen" menu entry; iOS Safari only offers the latter, under the Share
sheet). Installed, it opens in its own window with no browser chrome, gets a real home
screen/dock icon, and — since every feature runs client-side — it keeps working fully
offline after the first load: no server round-trip is needed to upload, extract, render,
or download anything.

### Browser requirements for rendering

Rendering (the final "draw the HUD onto the footage" step) depends on
[WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API) actually being
able to decode/encode your footage's codecs — upload, trimming, and telemetry extraction
don't need this and will always work. Two distinct things can break it, and the app will
warn you on the upload screen if either applies:

- **WebCodecs itself isn't available.** It's only exposed in a *secure context* (HTTPS, or
  the special-cased `localhost`/`127.0.0.1`) — a plain `http://` origin at a LAN or
  Tailscale address isn't one. See
  [Accessing it from other devices](#accessing-it-from-other-devices) for the two-minute fix.
- **WebCodecs is available, but this browser build can't decode H.264/AAC.** These are
  patent-licensed codecs, and action cams (GoPro included) almost always record in them.
  Many Linux distro-packaged Chromium/Chrome builds implement WebCodecs correctly but ship
  without licensed codec support at all — this looks identical to a real codec problem, but
  it's actually a browser build gap. Use the official Google Chrome or Microsoft Edge build
  instead (both bundle this codec support).

**HEVC (H.265) footage needs no special handling.** Chrome and Edge only expose HEVC when
the machine has a hardware decoder for it (VideoToolbox on macOS, the HEVC Video Extensions
on Windows, VAAPI on Linux), so plenty of otherwise-fine setups — a Linux desktop with an
NVIDIA or AMD GPU, a VM, a stock Windows install — can't decode the format newer GoPros
record in by default. Where the browser can't, the app decodes HEVC itself in software
using a WebAssembly build of [libde265](https://github.com/strukturag/libde265), and the
render proceeds normally. It's slower than hardware decoding (roughly 45fps at 1080p and
14fps at 4K, per CPU core), so long 4K clips take a while, and the render screen says so
while it's happening. Hardware decoding is always preferred when the browser has it.

**Rendered output is always 8-bit SDR.** If your camera records 10-bit or HDR (HLG/PQ) —
newer GoPros do at higher bitrates — the footage decodes fine, but the HUD is composited
through a 2D canvas, which is 8-bit sRGB, so the render is tone-mapped down to 8-bit on the
way out. The result is a normal, universally-playable file; it just won't carry HDR through.
This is a property of the compositing step, not of any one decoder, so it applies whether
the browser decoded the video or the software decoder did.

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
              lib/render/      the decode -> Canvas2D HUD draw -> encode -> mux pipeline,
                                run in workers/renderWorker.ts. Decoding uses WebCodecs;
                                hevcDecoder.ts supplies a WASM HEVC decoder for the many
                                browsers that can't decode it themselves
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
[libde265](https://github.com/strukturag/libde265) ·
[WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API)

libde265 is used via [`@yume-chan/libde265`](https://github.com/yume-chan/libde265), a
WebAssembly build of it, and is licensed under the LGPL-3.0 — compatible with this
project's AGPL-3.0. It ships as a separate `.wasm` module, fetched at runtime only when a
browser can't decode HEVC on its own.
