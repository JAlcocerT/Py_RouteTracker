"""GoPro embedded-metadata telemetry source.

Ported from legacy/overlay/racing_hud_v7.py:61-151 and legacy/overlay/lap_timer_v7.py:26-91.
Behavior is unchanged (same regexes, same GPMD ACCL struct layout, same
km/h conversion); what changed is that every knob is now a function argument
instead of a module-level constant, and paths are injected rather than
hardcoded to one developer's machine.
"""

from __future__ import annotations

import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.binaries import require_binary
from app.core.ffmpeg_utils import get_video_duration  # re-exported for callers that only import this module
from app.telemetry.resample import resample_to_grid
from app.telemetry.sources.base import TelemetryResult, empty_result

__all__ = [
    "dms_to_dd", "get_video_duration", "extract_exiftool_dump", "extract_gpmd_binary",
    "parse_gps_data", "parse_gpmd_accel", "sync_dataframes", "GoProEmbeddedSource",
]

_ENCODINGS = ("utf-8", "utf-16le", "latin-1")

_DMS_RE = re.compile(r"[deg'\"]+")
_SPEED_RE = re.compile(r"GPS Speed\s+:\s+([\d.]+)")
_LAT_RE = re.compile(r"GPS Latitude\s+:\s+(.+)")
_LON_RE = re.compile(r"GPS Longitude\s+:\s+(.+)")


def dms_to_dd(dms: str) -> float | None:
    """'37 deg 33' 30.60" N' -> 37.558500"""
    if not dms:
        return None
    try:
        parts = _DMS_RE.split(dms)
        dd = float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600
        if parts[3].strip() in ("S", "W"):
            dd *= -1
        return dd
    except (ValueError, IndexError):
        return None


def extract_exiftool_dump(video_path: Path, dest_txt: Path) -> Path:
    """Runs `exiftool -ee` once and writes the raw dump to dest_txt."""
    require_binary("exiftool")
    dest_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_txt, "w") as outfile:
        subprocess.run(["exiftool", "-ee", str(video_path)], stdout=outfile, check=True)
    return dest_txt


def extract_gpmd_binary(video_path: Path, dest_bin: Path) -> Path:
    """Dumps the raw GPMD (telemetry) data stream via ffmpeg's stream copy."""
    require_binary("ffmpeg")
    dest_bin.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:3", "-f", "data", "-c", "copy", str(dest_bin)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dest_bin


def parse_gps_data(txt_content: str, duration_sec: float) -> pd.DataFrame:
    """Parses an exiftool -ee text dump into a [time, lat, lon, speed] DataFrame.

    Speed is standardized to km/h (exiftool reports GPS Speed in m/s).
    """
    data: list[dict] = []
    cur_lat, cur_lon = np.nan, np.nan
    for line in txt_content.splitlines():
        m_spd = _SPEED_RE.search(line)
        if m_spd:
            data.append({"speed": float(m_spd.group(1)) * 3.6, "lat": cur_lat, "lon": cur_lon})
            continue
        m_lat = _LAT_RE.search(line)
        if m_lat:
            cur_lat = dms_to_dd(m_lat.group(1))
            continue
        m_lon = _LON_RE.search(line)
        if m_lon:
            cur_lon = dms_to_dd(m_lon.group(1))

    if not data:
        return pd.DataFrame(columns=["time", "lat", "lon", "speed"])

    df = pd.DataFrame(data)
    df[["lat", "lon"]] = df[["lat", "lon"]].ffill().bfill()
    df = df[(df["lat"] != 0) & (df["lon"] != 0)].reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame(columns=["time", "lat", "lon", "speed"])
    df["time"] = np.linspace(0, duration_sec, len(df))
    return df[["time", "lat", "lon", "speed"]]


def parse_gpmd_accel(binary_content: bytes, duration_sec: float, smoothing_window: int = 15) -> pd.DataFrame:
    """Parses raw GPMD ACCL samples out of the ffmpeg-dumped data stream.

    GPMD is a simple TLV stream: a 4-byte tag ('ACCL'), a type/size byte,
    a big-endian uint16 repeat count, then `repeat` samples of 3 big-endian
    int16s (in the camera's own accelerometer axes), padded to a 4-byte
    boundary. We only care about the ACCL tag here.
    """
    pts: list[tuple[int, int, int]] = []
    i = 0
    length = len(binary_content)
    while i < length - 8:
        if binary_content[i:i + 4] == b"ACCL":
            try:
                esize = binary_content[i + 5]
                repeat = struct.unpack(">H", binary_content[i + 6:i + 8])[0]
                pstart = i + 8
                total = esize * repeat
                pad = (total + 3) & ~3
                for k in range(repeat):
                    o = pstart + k * esize
                    val = struct.unpack(">hhh", binary_content[o:o + 6])
                    pts.append(val)
                i += 8 + pad
                continue
            except (struct.error, IndexError):
                pass
        i += 1

    df = pd.DataFrame(pts, columns=["c1", "c2", "c3"])
    if df.empty:
        return pd.DataFrame(columns=["time", "lat_g", "lon_g"])

    mag = np.sqrt(df["c1"] ** 2 + df["c2"] ** 2 + df["c3"] ** 2)
    one_g = mag.median()
    if not one_g:
        return pd.DataFrame(columns=["time", "lat_g", "lon_g"])

    df["lat_g"] = (df["c2"] / one_g).rolling(smoothing_window, center=True).mean().fillna(0)
    df["lon_g"] = (df["c3"] / one_g).rolling(smoothing_window, center=True).mean().fillna(0)
    df["time"] = np.linspace(0, duration_sec, len(df))
    return df[["time", "lat_g", "lon_g"]]


def sync_dataframes(df_gps: pd.DataFrame, df_accel: pd.DataFrame, duration_sec: float, target_fps: float) -> pd.DataFrame:
    """Resamples GPS (+ optional accel) onto one uniform `target_fps` time grid."""
    gps_re = resample_to_grid(df_gps, duration_sec, target_fps, ["lat", "lon", "speed"])

    if df_accel is not None and not df_accel.empty:
        accel_re = resample_to_grid(df_accel, duration_sec, target_fps, ["lat_g", "lon_g"])
        merged = pd.merge(gps_re, accel_re, on="time", how="left").fillna(0)
    else:
        merged = gps_re.copy()
        merged["lat_g"] = 0.0
        merged["lon_g"] = 0.0

    return merged[["time", "lat", "lon", "speed", "lat_g", "lon_g"]]


class GoProEmbeddedSource:
    """TelemetrySource backed by a GoPro's own embedded GPS + accelerometer track."""

    def __init__(self, cache_dir: Path, target_fps: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.target_fps = target_fps

    def extract(self, video_path: Path, duration_sec: float) -> TelemetryResult:
        video_path = Path(video_path)
        stem = video_path.stem

        txt_path = self.cache_dir / f"{stem}_exiftool.txt"
        if not txt_path.exists():
            extract_exiftool_dump(video_path, txt_path)
        gps_df = parse_gps_data(txt_path.read_text(encoding="utf-8", errors="ignore"), duration_sec)

        if gps_df.empty:
            return empty_result("gopro_embedded")

        bin_path = self.cache_dir / f"{stem}_gpmd.bin"
        if not bin_path.exists():
            extract_gpmd_binary(video_path, bin_path)
        accel_df = pd.DataFrame(columns=["time", "lat_g", "lon_g"])
        has_accel = False
        if bin_path.exists() and bin_path.stat().st_size > 0:
            accel_df = parse_gpmd_accel(bin_path.read_bytes(), duration_sec)
            has_accel = not accel_df.empty

        merged = sync_dataframes(gps_df, accel_df, duration_sec, self.target_fps)
        return TelemetryResult(df=merged, source_name="gopro_embedded", has_accel=has_accel)
