"""Render orchestration: parallel HUD-frame rendering + ffmpeg compositing.

This is the module that fixes the two biggest problems the audit found in
overlay/racing_hud_v7.py:
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
    """End-to-end: trim source video -> window telemetry -> render HUD
    frames in parallel -> composite onto the trimmed footage.

    `annotated_telemetry` must be sampled on a uniform time grid (see
    app.telemetry.resample.resample_to_grid) and, if lap widgets are
    enabled, annotated by app.laps.detection.detect_laps.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "hud_frames"
    trimmed_path = work_dir / "trimmed.mp4"

    def report(fraction: float) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)))

    report(0.0)
    trim_video(source_video, trim_start, trim_end, trimmed_path)
    report(0.1)

    window = annotated_telemetry[
        (annotated_telemetry["time"] >= trim_start) & (annotated_telemetry["time"] <= trim_end)
    ].copy()
    window["time"] = window["time"] - trim_start
    window = window.reset_index(drop=True)

    windowed_lap_indices = _remap_lap_indices_to_window(annotated_telemetry, lap_indices, trim_start, window)

    def render_progress(frac: float) -> None:
        report(0.1 + 0.75 * frac)

    render_hud_frames(window, windowed_lap_indices, config, frames_dir, n_workers=n_workers, on_progress=render_progress)
    report(0.85)

    fps = 1 / window["time"].diff().median() if len(window) > 1 else 30.0
    overlay_png_sequence(trimmed_path, frames_dir, fps, output_path)
    report(1.0)

    return RenderResult(output_path=output_path, frame_count=len(window))
