# Telemetry Overlay Webapp

A local, self-hosted webapp for overlaying GPS/G-force telemetry HUDs onto action-cam
footage: drag in a video, trim the section you want, pick which telemetry widgets to draw
(speedometer, lap timer, G-G diagram, minimap, session graph), and download the result.

This replaces the one-off `overlay/racing_hud_v7.py` / `overlay/lap_timer_v7.py` scripts in
the repo root with a proper upload → configure → render pipeline, and automatically
composites the HUD onto your real footage (the old scripts only ever rendered a standalone
HUD clip and printed an `ffmpeg` command for you to run by hand).

## Run it

```sh
cd webapp
docker compose up --build
```

Then open http://localhost:7000. Uploaded videos and rendered output persist in the
`routetracker_data` Docker volume across restarts.

### Accessing it from other devices over Tailscale

The container publishes port 7000 on all of the host's network interfaces (not just
`127.0.0.1`), so once it's running you can also reach it from any other device on your
Tailscale network at `http://<this-machine's-tailscale-ip-or-MagicDNS-name>:7000` — no
extra Docker network configuration needed. If it's not reachable, check that your host's
firewall (e.g. `ufw`) allows inbound connections on port 7000 from the `tailscale0`
interface.

## Supported telemetry sources

- **GoPro embedded GPS** — reads the camera's own embedded GPS/accelerometer metadata
  (via `exiftool`/`ffmpeg`, no separate file needed).
- **Video + separate GPX file** — for any other action cam (DJI, Insta360, older GoPros) or
  when you have a phone/Garmin/Polar GPX track instead. You can optionally give the video's
  real-world start time to sync the two automatically.

## Development

See `backend/readme.md` and the frontend's own Vite setup (`frontend/`, `npm run dev`).
