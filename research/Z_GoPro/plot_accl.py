#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from typing import List, Tuple, Literal

try:
    import matplotlib.pyplot as plt
except Exception:
    print("ERROR: Missing dependency 'matplotlib'. Install with: pip install matplotlib", file=sys.stderr)
    raise


def read_xyz_csv(csv_path: str) -> Tuple[List[float], List[float], List[float], List[float], Tuple[str, str, str], Literal["accel", "speed"]]:
    t_s: List[float] = []
    c1: List[float] = []
    c2: List[float] = []
    c3: List[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if {"t_s", "ax", "ay", "az"}.issubset(fields):
            cols = ("ax", "ay", "az")
            kind: Literal["accel", "speed"] = "accel"
        elif {"t_s", "vx", "vy", "vz"}.issubset(fields):
            cols = ("vx", "vy", "vz")
            kind = "speed"
        else:
            raise RuntimeError(
                f"CSV missing required headers for accel or speed. Got: {reader.fieldnames}"
            )

        for row in reader:
            try:
                t_s.append(float(row["t_s"]))
                c1.append(float(row[cols[0]]))
                c2.append(float(row[cols[1]]))
                c3.append(float(row[cols[2]]))
            except Exception:
                # Skip malformed rows
                continue
    if not t_s:
        raise RuntimeError("No valid rows read from CSV")
    return t_s, c1, c2, c3, cols, kind


def plot_series(t_s: List[float], series: List[List[float]], labels: List[str], kind: Literal["accel", "speed"], title: str):
    plt.figure(figsize=(12, 6))
    for y, lab in zip(series, labels):
        plt.plot(t_s, y, label=lab)
    plt.xlabel("Time (s)")
    if kind == "accel":
        plt.ylabel("Acceleration (parser units)")
    else:
        plt.ylabel("Speed (accel_units*s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot CSV over time: accepts accelerometer (t_s, ax, ay, az) or speed (t_s, vx, vy, vz)")
    parser.add_argument("csv", help="Path to CSV (accelerometer from extract_accel.py or speed from integrate_accl_to_speed.py)")
    parser.add_argument("-o", "--output", help="Path to save the PNG (defaults next to CSV)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument("--axis", nargs="+", help="Subset of axes to plot, e.g. --axis ax ay or --axis vx vz")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        t_s, s1, s2, s3, labels, kind = read_xyz_csv(csv_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.abspath(args.output) if args.output else os.path.join(os.path.dirname(csv_path), f"{stem}_plot_accl.png")

    # Select which axes to plot
    data_map = {labels[0]: s1, labels[1]: s2, labels[2]: s3}
    if args.axis:
        selected = args.axis
    else:
        selected = list(labels)
    invalid = [a for a in selected if a not in data_map]
    if invalid:
        print(f"ERROR: Unknown axis names: {invalid}. Available: {list(labels)}", file=sys.stderr)
        sys.exit(2)
    series = [data_map[a] for a in selected]

    title = f"{stem} - {'Accelerometer' if kind == 'accel' else 'Speed'}"
    plot_series(t_s, series, selected, kind, title=title)

    try:
        plt.savefig(out_path, dpi=150)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"ERROR saving figure: {e}", file=sys.stderr)
        sys.exit(3)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
