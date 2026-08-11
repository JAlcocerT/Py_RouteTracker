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

## Storage lifecycle

Two independent, always-on cleanup mechanisms (see root `README.md` for the user-facing
version) — neither is optional, both are covered by `tests/test_cleanup.py` and
`tests/test_api.py`:

1. **Per-render scratch** (`work_dir`: the trimmed clip + HUD frame PNGs + telemetry
   manifest) is deleted immediately after a render finishes or fails -- in
   `LocalRenderWorker._process_one` for a local render, or in the `/complete`/`/fail`
   handlers of `routes_worker.py` for a remote one. Cleanup always happens *before* the job
   is marked `done`/`error`, not after -- otherwise a client polling status could observe
   "done" and check for the (still briefly present) work_dir in the gap before cleanup runs.
2. **Uploaded videos, GPX files, cached telemetry/lap parquet, and rendered output** are
   deleted by `app.core.cleanup.RetentionSweeper`, a daemon thread started in `main.py`'s
   lifespan handler that runs `sweep_expired_videos` every `ROUTETRACKER_SWEEP_INTERVAL_SECONDS`
   (default 300s). A video is deleted once `ROUTETRACKER_RETENTION_MINUTES` (default 60) have
   passed since `VideoMeta.created_at` — *unless* its extraction job or any of its render jobs
   (`VideoMeta.render_job_ids`) is still `pending`/`running`, in which case it's left alone and
   retried on the next sweep, so a slow render's files are never pulled out from under it.

`GET /api/health` reports the current `retention_minutes`, and `GET /api/videos/{id}` reports
that video's computed `expires_at` — the frontend uses both to tell users when their file will
be gone.

## Distributed rendering

See root `README.md`'s "Distributed rendering" section for the user-facing picture and
security model. Architecture:

- **`app/render/video_render.py`** splits the render at its natural cheap/expensive
  boundary: `prepare_render_job` (trim + window telemetry -- cheap, always runs on the
  coordinator, since only it has the original upload) produces a `PreparedRenderJob`;
  `execute_prepared_render` (draw HUD frames + composite -- the actual bottleneck) runs
  identically whether called in-process or by a worker that fetched its inputs over HTTP.
  `render_and_composite` is kept as a thin wrapper calling both, for callers (tests, and
  conceptually "a single machine doing everything") that don't care about the split.
- **`app/core/jobs.py`**'s `jobs` table gained `payload` (the JSON render request),
  `worker_id`, `claim_token`, and `released` columns (migrated via a guarded `ALTER TABLE`,
  not a fresh-schema assumption). `enqueue`/`claim_next`/`claim_specific`/`release`/
  `requeue_stale` implement a pull-based claim queue -- workers ask for work rather than the
  coordinator tracking who's available. `claim_next` claims the oldest `pending`, *released*
  job of a kind (used by standing workers, and the built-in local worker); `claim_specific`
  claims one exact job by id, only if still `pending` -- ignoring `released` entirely (used
  by job-scoped self-render). Both do their select-then-update under the manager's existing
  lock, so concurrent claimers (the local worker, any standing remote ones, and a self-render
  attempt, all at once) can never claim the same job twice. Only render jobs use any of this;
  extraction jobs still go straight through the older `create_job`/`submit` (immediate,
  in-process, unchanged).

  A freshly enqueued render job starts with `released = 0` -- invisible to `claim_next`
  (neither the built-in local worker nor any standing remote worker will touch it) until
  either the uploader self-renders it via `claim_specific` (no release needed, works
  immediately), or explicitly calls `release()` (see `POST /api/jobs/{id}/release` in
  `routes_jobs.py`, wired to the "Render on the server instead" button in the UI). There is
  no timeout of any kind here -- an earlier version auto-released jobs after a fixed grace
  period (`ROUTETRACKER_SELF_RENDER_GRACE_SECONDS`), which meant the local worker's own ~2s
  poll loop could silently start rendering before a human had actually decided anything; this
  was replaced with waiting indefinitely for an explicit decision after watching that
  countdown confuse exactly the person it was meant to help. `requeue_stale` does not reset
  `released` -- a job that already got explicitly released and then lost its worker (crash,
  network drop) goes back to being claimable without asking the uploader to decide again.
- **`app/render/coordinator.py`**'s `_prepare_claimed_job` is the shared "load telemetry +
  trim + window + write manifest" step both `claim_and_prepare_render` (next-in-queue) and
  `claim_and_prepare_specific_render` (one exact job, for self-render) call after their
  respective claim. Writes a `telemetry.json` manifest alongside the trimmed clip in
  `work_dir` (`settings.work_dir / job_id`, a fixed convention so a later, separate HTTP
  request -- a worker fetching inputs -- can find them with no in-memory state to carry
  over).
- **`app/render/local_worker.py`**'s `LocalRenderWorker` is a background thread (started in
  `main.py`'s lifespan, same pattern as `RetentionSweeper`) that loops
  `claim_and_prepare_render` + `execute_prepared_render` -- this is what makes local-only
  operation work with zero configuration; it's simply always the first available worker.
  Deliberately processes one job at a time (unlike the old `ThreadPoolExecutor` submission
  path, which could run several renders concurrently). The same file's
  `StaleRenderJobRequeuer` periodically calls `requeue_stale` so a render claimed by a worker
  that vanished mid-job (closed laptop, network drop) doesn't stay stuck forever --
  `ROUTETRACKER_WORKER_LEASE_MINUTES` (default 30).
- **`app/api/routes_worker.py`**, prefix `/api/worker`, has two auth dependencies:
  `require_global_worker_token` (the admin's `ROUTETRACKER_WORKER_TOKEN`; 503 if unset, 401
  on mismatch) guards only `GET /jobs/next` (claim-whatever's-oldest only makes sense for a
  trusted standing worker). `require_job_access` guards everything else once a job_id is in
  scope (`POST /jobs/{id}/claim`, `GET /jobs/{id}/inputs/{video,telemetry}`, `POST
  /jobs/{id}/progress`, `POST /jobs/{id}/complete`, `POST /jobs/{id}/fail`) -- it accepts
  *either* the global token *or* that specific job's `claim_token`, so the self-render path
  works even when `ROUTETRACKER_WORKER_TOKEN` is never set at all.
- **`app/worker_main.py`** (`python -m app.worker_main --server ... --token ...`) is the
  standalone worker CLI. Deliberately never imports `app.core.config`/`settings` at all --
  it's `httpx` calls to the coordinator plus the pure rendering functions above, into its
  own `tempfile.TemporaryDirectory`. Two modes: the default polling loop (`run_worker`,
  claims via `/jobs/next`, needs the global token) and `--job <id>` (`run_single_job`, one
  `POST /jobs/{id}/claim` then exit -- a 409 there means something else already has it,
  which is a clean no-op, not an error). Runs from the same Docker image already published
  by CI/CD (`docker-multiarch.yml`), just with a different `command:`.

Tests: `tests/test_jobs.py` covers both claim paths directly (atomicity, kind filtering,
requeue, that `claim_specific` doesn't disturb `claim_next`'s ordering for other queued jobs,
that `claim_next` ignores unreleased jobs while `claim_specific` ignores `released` entirely,
and that `release()` is what makes a job visible to `claim_next`).
`tests/test_video_render.py` covers the prepare/execute split producing equivalent
output to the combined pipeline. `tests/test_routes_worker.py` covers the full worker HTTP
API using a second, lifespan-free FastAPI app (so `LocalRenderWorker` isn't running in the
background racing the test's own manual claims) -- auth/feature-flag behavior for both
dependencies, cross-job token isolation (job A's token can't touch job B), and two full
round trips (claim → fetch inputs → report progress → complete): one via the global token,
one via only a job's own `claim_token` with `ROUTETRACKER_WORKER_TOKEN` unset entirely. Both
paths were also manually verified against real, separate `app.worker_main` processes
(different CWD, no shared environment with the coordinator) claiming and completing real
renders over HTTP -- including confirming a `--job` self-render completes with the
coordinator's admin token never configured at all.
