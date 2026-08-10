# Backend

FastAPI service for the telemetry-overlay pipeline: upload → extract → configure → render.
Ported from the prototype scripts now archived at `../legacy/overlay/`. See the root
`README.md` for the product-level picture.

## Requirements

Runtime needs the `ffmpeg`, `ffprobe`, and `exiftool` binaries on `PATH` (the
Docker image installs them via apt; `app.core.binaries` fails fast with a
clear error if they're missing rather than crashing deep in a subprocess call).

## Local development

```sh
uv sync
uv run uvicorn app.main:app --reload --port 7000
```

## Tests

```sh
uv run pytest
```

Most tests need no external binaries — telemetry parsing, lap detection,
extrema comparison, the job manager, and the HUD drawing layer are all
tested against fixtures committed in `tests/fixtures/` (copied from the
`research/` sample data, kept independent of it so the test suite doesn't
depend on that folder's contents) or synthetic data.

A handful of tests that actually invoke `ffmpeg` (trimming, PNG-sequence
compositing, and the full render-job lifecycle) are skipped automatically
when `ffmpeg` isn't on `PATH` — which is the case on a bare dev machine
without root/apt access. They run for real:
- inside the Docker image (`ffmpeg`/`exiftool` are apt-installed there), or
- locally, if you put a static `ffmpeg`/`ffprobe` build on `PATH` yourself
  (e.g. via the `static-ffmpeg` PyPI package, which needs no root/apt access)
  before running `pytest`.

The full upload → extract → detect-laps → compare-laps → render → composite
→ download pipeline has been manually verified both against a locally
installed static ffmpeg build and inside the built Docker image, using a
synthetic `ffmpeg -f lavfi testsrc` video and the sample GPX fixture in
`tests/fixtures/`.

## Rendering performance

Two independent levers, tuned for running on modest/shared hardware (e.g. a non-dedicated
homelab box) without touching visual quality at all — same resolution, same DPI, same fonts,
same glow effects, just less redundant work per frame:

1. **Real blitting** (`app/render/hud_layers.py`). The static parts of the HUD (axes,
   background lines, glow effects) are drawn and cached once via `copy_from_bbox`; each frame
   only restores that cached region and redraws the handful of artists that actually changed
   (`draw_artist` + `blit`), then the canvas' raw RGBA buffer is saved directly instead of
   going through `Figure.savefig` (which redoes a full figure layout every frame — the actual
   reason the original prototype scripts were so slow). Measured on this dev machine at full
   production settings (1600×900, DPI 100, all 4 widgets): **90.7ms → 54.8ms per frame
   (~1.65x)**, pixel-for-pixel identical output.
2. **Multiprocessing** across frame chunks (`app/render/video_render.py`), already present
   before this round of tuning. Speedup is real but sub-linear in practice — each worker is a
   fresh Python + matplotlib process (`spawn`, deliberately not `fork`, since the job manager
   itself is multi-threaded and fork+threads risk deadlocks), so there's fixed per-worker
   startup cost that amortizes better over longer clips. Measured locally (4 cores, 900
   frames): 1 worker 17.5fps, 4 workers 28.1fps.

Tuning knobs (env vars, both have safe defaults):
- `ROUTETRACKER_MAX_RENDER_WORKERS` — defaults to `min(4, cpu_count - 1)`, deliberately capped
  rather than defaulting to every-core-but-one, since each worker's ~150-250MB baseline
  footprint adds up fast on a constrained/shared host. Raise it if you know you have the
  headroom.
- `ROUTETRACKER_TARGET_FPS` — defaults to 30. Not changed by this round of tuning (lowering it
  would reduce render workload further but was left alone deliberately, since it trades
  smoothness for speed rather than being a free win).
