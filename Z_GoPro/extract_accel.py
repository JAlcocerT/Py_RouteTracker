#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Tuple

try:
    from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
except Exception as e:
    print("ERROR: Missing dependency 'py-gpmf-parser'. Install with: pip install --user py-gpmf-parser", file=sys.stderr)
    raise


def extract_accl_to_csv(input_mp4: str, output_csv: str) -> None:
    extractor = GoProTelemetryExtractor(input_mp4)
    extractor.open_source()
    try:
        accl, accl_t = extractor.extract_data("ACCL")
    finally:
        extractor.close_source()

    if accl is None or accl_t is None:
        raise RuntimeError("No ACCL data found in the file.")

    # Expect accl shape: (N, 3) for X, Y, Z and accl_t shape: (N,)
    if len(accl) != len(accl_t):
        raise RuntimeError(f"Mismatched lengths: ACCL={len(accl)} timestamps={len(accl_t)}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "ax", "ay", "az"])  # time in seconds, acceleration units as provided by parser
        for t, vec in zip(accl_t, accl):
            # vec expected iterable of length 3
            if isinstance(vec, (list, tuple)) and len(vec) >= 3:
                ax, ay, az = vec[0], vec[1], vec[2]
            else:
                # Fallback in case of different structure
                try:
                    ax, ay, az = vec
                except Exception:
                    raise RuntimeError("Unexpected ACCL vector format; expected length-3 values per sample.")
            writer.writerow([t, ax, ay, az])

    print(f"Wrote accelerometer CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract GoPro accelerometer (ACCL) to CSV")
    parser.add_argument("-i", "--input", required=True, help="Path to GoPro MP4/MOV file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: script_dir/<video_stem>_accel.csv)")
    args = parser.parse_args()

    input_mp4 = os.path.abspath(args.input)
    if not os.path.isfile(input_mp4):
        print(f"ERROR: Input file not found: {input_mp4}", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_stem = os.path.splitext(os.path.basename(input_mp4))[0]
    default_out = os.path.join(script_dir, f"{video_stem}_accel.csv")
    output_csv = os.path.abspath(args.output) if args.output else default_out

    try:
        extract_accl_to_csv(input_mp4, output_csv)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
