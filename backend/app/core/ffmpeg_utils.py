"""Thin wrappers around the ffmpeg/ffprobe CLI calls the pipeline needs.

Kept deliberately dumb (no retries) — subprocess calls to a well-known CLI,
not a library to abstract over. The one exception is `trim_video` and
`overlay_png_sequence`'s optional `on_progress`: both run as a single,
long, blocking ffmpeg call on a large video, and that call is also the only
heartbeat for the render job's claim lease (see
app.render.local_worker.StaleRenderJobRequeuer) -- with no progress parsing
at all, a big-enough source file silently starves the heartbeat and the job
gets reclaimed as "stale" out from under a worker that's still actively
running it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from app.core.binaries import require_binary

ProgressCallback = Callable[[float], None]


def _run_ffmpeg_with_heartbeat(cmd: list[str], total_duration: float, on_progress: ProgressCallback | None) -> None:
    """Runs an ffmpeg command, optionally reporting fractional progress by
    reading ffmpeg's own `-progress pipe:1` machine-readable output instead
    of polling from outside. Falls back to a plain blocking `subprocess.run`
    when there's no callback (or no known duration to compute a fraction
    against) -- most ffprobe/ffmpeg call sites don't need this."""
    if on_progress is None or total_duration <= 0:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return

    # Global options, so position relative to the rest of `cmd` doesn't
    # matter. out_time_us is genuinely microseconds; out_time_ms has a
    # long-standing ffmpeg quirk where it's actually also microseconds, not
    # milliseconds, so it's avoided here.
    cmd_with_progress = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    proc = subprocess.Popen(cmd_with_progress, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key != "out_time_us":
                continue
            try:
                on_progress(min(1.0, max(0.0, int(value) / 1_000_000 / total_duration)))
            except ValueError:
                continue
    finally:
        proc.stdout.close()
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd_with_progress)


def get_video_duration(video_path: Path) -> float:
    require_binary("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def get_video_fps(video_path: Path) -> float:
    """Real frame rate of the video's first video stream (e.g. 29.97 for a
    30000/1001 NTSC-style rate). Used to default telemetry resampling to the
    footage's own rate instead of a flat constant -- see
    app.api.routes_videos.upload_video."""
    require_binary("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """(width, height) of the video's first stream, in pixels."""
    require_binary("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    width_str, height_str = out.split("x")
    return int(width_str), int(height_str)


def trim_video(source: Path, start_sec: float, end_sec: float, dest: Path, on_progress: ProgressCallback | None = None) -> Path:
    """Frame-accurate trim via re-encode (stream-copy trims only cut on
    keyframes, which would desync the HUD overlay from the footage)."""
    require_binary("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        str(dest),
    ]
    _run_ffmpeg_with_heartbeat(cmd, duration, on_progress)
    return dest


def overlay_png_sequence(
    trimmed_video: Path,
    frames_dir: Path,
    fps: float,
    output_path: Path,
    frame_pattern: str = "frame_%06d.png",
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Composites a transparent HUD PNG sequence onto a video via ffmpeg's
    `overlay` filter. This is the step legacy/overlay/racing_hud_v7.py never
    actually did — it rendered the HUD as a standalone clip and printed an
    ffmpeg command for the user to run by hand."""
    require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(trimmed_video),
        "-framerate", str(fps),
        "-i", str(frames_dir / frame_pattern),
        "-filter_complex", "[0:v][1:v]overlay=format=auto[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        str(output_path),
    ]
    # trimmed_video's own duration, not the (possibly much longer) original
    # source -- overlay_png_sequence always runs on the already-trimmed clip.
    total_duration = get_video_duration(trimmed_video) if on_progress else 0.0
    _run_ffmpeg_with_heartbeat(cmd, total_duration, on_progress)
    return output_path
