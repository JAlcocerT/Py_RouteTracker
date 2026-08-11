"""Render orchestration: parallel HUD-frame rendering + ffmpeg compositing.

This is the module that fixes the two biggest problems the audit found in
legacy/overlay/racing_hud_v7.py:
  1. Rendering was single-threaded matplotlib (the repo's own comparison.md
     logs a 14+ minute render for one clip) -- here frames are split across
     worker processes.
  2. v7 never composited the HUD onto the real footage; it produced a
     separate clip and printed an ffmpeg command for the user to run by
     hand. `render_and_composite` does the trim + render + overlay as one
     automatic pipeline.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.core.ffmpeg_utils import overlay_png_sequence, trim_video
from app.render.hud_layers import HudRenderer, RenderConfig

ProgressCallback = Callable[[float], None]

# Populated per-worker-process by _init_worker; multiprocessing.Pool workers
# each get their own copy via fork/spawn, so this is not shared mutable state.
_worker_state: tuple | None = None


def _init_worker(df: pd.DataFrame, lap_indices: list[int], config: RenderConfig, frames_dir: Path, progress_queue) -> None:
    global _worker_state
    _worker_state = (df, lap_indices, config, frames_dir, progress_queue)


def _render_chunk_worker(frame_indices: list[int]) -> int:
    df, lap_indices, config, frames_dir, progress_queue = _worker_state
    renderer = HudRenderer(df, lap_indices, config)
    try:
        for f in frame_indices:
            renderer.draw_frame(f)
            renderer.save_frame(frames_dir / f"frame_{f:06d}.png")
            progress_queue.put(1)
    finally:
        renderer.close()
    return len(frame_indices)


def render_hud_frames(
    df: pd.DataFrame,
    lap_indices: list[int],
    config: RenderConfig,
    frames_dir: Path,
    n_workers: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Renders one transparent PNG per row of `df` into `frames_dir`,
    splitting the work across `n_workers` processes. Returns frame count."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(df)
    if total == 0:
        return 0

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers, total))
    chunks = [c.tolist() for c in np.array_split(np.arange(total), n_workers) if len(c) > 0]

    # Use "spawn" rather than the platform default "fork": this pool is
    # started from inside a background thread (the job manager), and
    # fork()-ing a multi-threaded process risks deadlocks in the child.
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()

    with ctx.Pool(
        processes=len(chunks),
        initializer=_init_worker,
        initargs=(df, lap_indices, config, frames_dir, progress_queue),
    ) as pool:
        async_result = pool.map_async(_render_chunk_worker, chunks)
        completed = 0
        while completed < total:
            try:
                progress_queue.get(timeout=0.5)
                completed += 1
                if on_progress:
                    on_progress(completed / total)
            except queue_mod.Empty:
                if async_result.ready():
                    break
        async_result.get()

    return total


def _remap_lap_indices_to_window(full_df: pd.DataFrame, lap_indices: list[int], trim_start: float, windowed_df: pd.DataFrame) -> list[int]:
    """lap_indices are positional indices into the full-session telemetry.
    After slicing to [trim_start, trim_end] and resetting time to 0, those
    positions no longer line up -- remap each crossing by its absolute time
    instead, dropping crossings outside the render window."""
    if windowed_df.empty:
        return []
    remapped = []
    for i in lap_indices:
        if i >= len(full_df):
            continue
        crossing_time = full_df.iloc[i]["time"] - trim_start
        if crossing_time < windowed_df["time"].min() or crossing_time > windowed_df["time"].max():
            continue
        pos = int((windowed_df["time"] - crossing_time).abs().idxmin())
        remapped.append(pos)
    return remapped


@dataclass
class RenderResult:
    output_path: Path
    frame_count: int


@dataclass
class PreparedRenderJob:
    """The output of the cheap, coordinator-side half of a render: a
    trimmed clip + the small windowed telemetry slice needed to draw the
    HUD over it. This is the natural hand-off point for a distributed
    worker (see app.render.local_worker / app.worker_main) -- it's
    everything `execute_prepared_render` needs and nothing more, so a
    remote worker only ever has to transfer this (typically short) trimmed
    clip and a small JSON telemetry payload, never the full, potentially
    multi-GB, original upload."""

    trimmed_video_path: Path
    windowed_telemetry: pd.DataFrame
    lap_indices: list[int]
    fps: float
    work_dir: Path


def prepare_render_job(
    source_video: Path,
    annotated_telemetry: pd.DataFrame,
    lap_indices: list[int],
    trim_start: float,
    trim_end: float,
    work_dir: Path,
) -> PreparedRenderJob:
    """The cheap half of a render: trim the source video and window the
    telemetry to the requested [trim_start, trim_end] range. Always runs on
    the coordinator (it has the original upload; a worker never needs to).

    `annotated_telemetry` must be sampled on a uniform time grid (see
    app.telemetry.resample.resample_to_grid) and, if lap widgets are
    enabled, annotated by app.laps.detection.detect_laps.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    trimmed_path = work_dir / "trimmed.mp4"
    trim_video(source_video, trim_start, trim_end, trimmed_path)

    window = annotated_telemetry[
        (annotated_telemetry["time"] >= trim_start) & (annotated_telemetry["time"] <= trim_end)
    ].copy()
    window["time"] = window["time"] - trim_start
    window = window.reset_index(drop=True)

    windowed_lap_indices = _remap_lap_indices_to_window(annotated_telemetry, lap_indices, trim_start, window)
    fps = 1 / window["time"].diff().median() if len(window) > 1 else 30.0

    return PreparedRenderJob(
        trimmed_video_path=trimmed_path,
        windowed_telemetry=window,
        lap_indices=windowed_lap_indices,
        fps=fps,
        work_dir=work_dir,
    )


def execute_prepared_render(
    prepared: PreparedRenderJob,
    config: RenderConfig,
    output_path: Path,
    n_workers: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
    """The expensive half of a render: draw the HUD frames in parallel and
    composite them onto the already-trimmed clip. This is the part that
    gets distributed -- runs identically whether called locally
    (app.render.local_worker) or from a remote worker process
    (app.worker_main) that downloaded `prepared`'s inputs over HTTP."""
    frames_dir = prepared.work_dir / "hud_frames"

    def report(fraction: float) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)))

    report(0.0)
    render_hud_frames(
        prepared.windowed_telemetry, prepared.lap_indices, config, frames_dir,
        n_workers=n_workers, on_progress=lambda f: report(0.9 * f),
    )
    report(0.9)

    overlay_png_sequence(prepared.trimmed_video_path, frames_dir, prepared.fps, output_path)
    report(1.0)

    return RenderResult(output_path=output_path, frame_count=len(prepared.windowed_telemetry))


def render_and_composite(
    source_video: Path,
    annotated_telemetry: pd.DataFrame,
    lap_indices: list[int],
    config: RenderConfig,
    trim_start: float,
    trim_end: float,
    work_dir: Path,
    output_path: Path,
    n_workers: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
    """The full, single-machine pipeline: `prepare_render_job` then
    `execute_prepared_render` back to back. Kept as a convenience wrapper
    -- callers that don't care about distributing the work (tests, the
    local worker) can use this one function exactly as before."""

    def report(fraction: float) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)))

    report(0.0)
    prepared = prepare_render_job(source_video, annotated_telemetry, lap_indices, trim_start, trim_end, work_dir)
    report(0.1)

    return execute_prepared_render(
        prepared, config, output_path, n_workers=n_workers,
        on_progress=lambda f: report(0.1 + 0.9 * f),
    )
