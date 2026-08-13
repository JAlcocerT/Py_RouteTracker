"""Diagnostic: prints the raw `GPS Speed` numbers ExifTool reports for a
GoPro clip, before any unit conversion, so you can tell by eye whether they
look like m/s (roughly top-speed-kmh / 3.6) or already like km/h.

Usage:
    python backend/scripts/inspect_gps_speed.py path/to/clip.mp4
    python backend/scripts/inspect_gps_speed.py path/to/already_dumped.txt

Needs `exiftool` on PATH when given a video file directly.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.telemetry.sources.gopro_embedded import _SPEED_RE, convert_speed_to_kmh  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.suffix.lower() == ".txt":
        content = path.read_text(encoding="utf-8", errors="ignore")
    else:
        print(f"Running `exiftool -ee` on {path} ...")
        content = subprocess.run(
            ["exiftool", "-ee", str(path)], capture_output=True, text=True, check=True
        ).stdout

    values: list[float] = []
    units: Counter[str] = Counter()
    for line in content.splitlines():
        m = _SPEED_RE.search(line)
        if m:
            value, unit = m.groups()
            values.append(float(value))
            units[unit or "(none)"] += 1

    if not values:
        print("No `GPS Speed` lines found -- is this really a GoPro exiftool -ee dump?")
        sys.exit(1)

    converted = []
    for line in content.splitlines():
        m = _SPEED_RE.search(line)
        if m:
            value, unit = m.groups()
            converted.append(convert_speed_to_kmh(float(value), unit))

    print(f"\n{len(values)} GPS Speed samples found")
    print(f"unit tokens seen: {dict(units)}")
    print(f"\nRAW values (as printed by exiftool, before any conversion):")
    print(f"  min={min(values):.3f}  max={max(values):.3f}  mean={sum(values)/len(values):.3f}")
    print(f"\nAFTER current code's conversion (what the app would show, in km/h):")
    print(f"  min={min(converted):.3f}  max={max(converted):.3f}  mean={sum(converted)/len(converted):.3f}")

    print(
        "\nSanity check: if the RAW max looks like a plausible speed in km/h "
        "ALREADY (e.g. ~80-120 for a car/kart, ~30-60 for a bike) and the unit "
        "tokens are all '(none)', the code is almost certainly over-converting "
        "-- that raw number is not m/s, so multiplying by 3.6 again inflates it. "
        "If the RAW max instead looks like a plausible speed in m/s (i.e. "
        "roughly the expected km/h top speed divided by 3.6), the current "
        "conversion is correct."
    )


if __name__ == "__main__":
    main()
