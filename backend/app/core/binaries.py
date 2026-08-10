"""Guards around the external CLI tools the whole pipeline depends on.

None of the original overlay/*.py scripts ever checked these were installed —
they'd just fail deep inside a subprocess call with a cryptic error. Since the
webapp runs unattended (a background job, not someone watching a terminal),
missing binaries need to fail fast and legibly.
"""

from __future__ import annotations

import shutil

REQUIRED_BINARIES = ("ffmpeg", "ffprobe", "exiftool")


class MissingBinaryError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(
            f"Required binary '{name}' was not found on PATH. "
            f"Install it (the Docker image installs {', '.join(REQUIRED_BINARIES)} "
            "via apt) before using the telemetry/render pipeline."
        )
        self.name = name


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MissingBinaryError(name)
    return path


def check_all_present() -> None:
    for name in REQUIRED_BINARIES:
        require_binary(name)
