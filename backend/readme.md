# Backend

FastAPI service that ports and parameterizes the logic from
`overlay/racing_hud_v7.py` and `overlay/lap_timer_v7.py` into a proper
upload → extract → configure → render pipeline. See `webapp/README.md` for
the product-level picture and `../../.claude/plans/scalable-meandering-lecun.md`
(if still present) for the original design plan.

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
tested against fixtures already committed in the repo
(`Z_GoPro/*_telemetry.txt`, `Data_My_Routes/*.gpx`) or synthetic data.

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
synthetic `ffmpeg -f lavfi testsrc` video and the sample GPX fixture under
`Data_My_Routes/`.
