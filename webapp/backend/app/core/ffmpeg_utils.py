"""Thin wrappers around the ffmpeg/ffprobe CLI calls the pipeline needs.

Kept deliberately dumb (no retries, no progress parsing) — subprocess calls
to a well-known CLI, not a library to abstract over.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.binaries import require_binary


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


def trim_video(source: Path, start_sec: float, end_sec: float, dest: Path) -> Path:
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
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return dest


def overlay_png_sequence(
    trimmed_video: Path,
    frames_dir: Path,
    fps: float,
    output_path: Path,
    frame_pattern: str = "frame_%06d.png",
) -> Path:
    """Composites a transparent HUD PNG sequence onto a video via ffmpeg's
    `overlay` filter. This is the step overlay/racing_hud_v7.py never
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
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path
