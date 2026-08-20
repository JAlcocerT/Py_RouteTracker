"""Loss-less joining of split action-cam recordings.

Some action cams (GoPro chaptered recordings above the ~4GB FAT32/exFAT
boundary, and others) split one continuous recording across multiple files.
Naively concatenating them in a video editor -- or via any re-encode --
typically drops the embedded GPMF telemetry data stream that
app.telemetry.sources.gopro_embedded reads (it hardcodes stream index
`0:3` for that track). This module joins parts at the container level
instead, via ffmpeg's own concat demuxer doing a full stream copy: no
re-encode, and every stream -- including gpmd -- survives verbatim at its
original index, so the existing extraction code needs no changes to read
a joined file.

Deliberately doesn't reach for a dedicated tool like GPAC's MP4Box (a
common recommendation for exactly this) or the third-party gopro-overlay
package's `gopro-join` -- both work, but both are a new dependency this app
doesn't otherwise need, when ffmpeg (already required by
app.core.binaries) does the job.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.binaries import require_binary


class IncompatiblePartsError(ValueError):
    """Raised when video parts don't share the stream layout a lossless
    container-level join requires."""


def probe_stream_signature(video_path: Path) -> dict:
    """A small ffprobe fingerprint (codec/resolution/frame rate) of a
    part's primary video stream. Used to catch parts that aren't actually
    continuations of the same recording before attempting a join --
    ffmpeg's concat demuxer stream-copy either fails deep in a subprocess
    call or silently produces a broken file when codec parameters differ
    between parts, both worse failure modes than rejecting it up front
    with a clear message.
    """
    require_binary("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    codec, width, height, rate = out.split(",")
    return {"codec": codec, "width": int(width), "height": int(height), "fps": rate}


def validate_join_compatible(parts: list[Path]) -> None:
    """Raises IncompatiblePartsError if the given parts don't share the
    same video codec/resolution/frame rate as the first part -- true for
    chaptered recordings from the same camera session, not for unrelated
    clips someone accidentally selected together.
    """
    if len(parts) < 2:
        raise IncompatiblePartsError("need at least two video parts to join")
    signatures = [probe_stream_signature(p) for p in parts]
    first = signatures[0]
    for path, sig in zip(parts[1:], signatures[1:]):
        if sig != first:
            raise IncompatiblePartsError(
                f"'{path.name}' doesn't match the first part's format "
                f"({sig} vs {first}) -- a lossless join needs every part "
                "to share the same codec, resolution, and frame rate. "
                "This usually means these files aren't chapters of the "
                "same recording, or one of them has been re-encoded."
            )


def _concat_list_line(path: Path) -> str:
    # ffmpeg's concat-demuxer list format takes single-quoted paths; a
    # literal single quote in the path has to be escaped as '\'' (close
    # the quote, an escaped quote, reopen the quote) rather than assumed
    # absent -- upload filenames are arbitrary user input.
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def join_videos(parts: list[Path], dest: Path) -> Path:
    """Losslessly concatenates already-validated-compatible video parts
    into dest, preserving every stream -- including a GoPro's embedded
    gpmd telemetry track -- at its original index.

    `-map 0` (rather than ffmpeg's default per-type "best stream" auto
    selection, which only picks one video/audio stream and drops data
    streams entirely) is what pulls the telemetry data stream into the
    output at all; `-copy_unknown` is needed on top of that because
    ffmpeg doesn't recognize GPMF's `gpmd` codec tag and otherwise
    refuses to copy a stream type it can't identify.
    """
    require_binary("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)

    list_path = dest.parent / f"{dest.stem}_parts.txt"
    list_path.write_text("".join(_concat_list_line(p) for p in parts))
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-map", "0", "-c", "copy", "-copy_unknown",
            str(dest),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        list_path.unlink(missing_ok=True)
    return dest
