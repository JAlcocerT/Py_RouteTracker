#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    print("ERROR: Missing dependency 'matplotlib'. Install with: pip install matplotlib", file=sys.stderr)
    raise


def read_accl_csv(csv_path: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    t_s: List[float] = []
    ax: List[float] = []
    ay: List[float] = []
    az: List[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"t_s", "ax", "ay", "az"}
        if not required.issubset(reader.fieldnames or {}):
            raise RuntimeError(
                f"CSV missing required headers {required}. Got: {reader.fieldnames}"
            )
        for row in reader:
            try:
                t_s.append(float(row["t_s"]))
                ax.append(float(row["ax"]))
                ay.append(float(row["ay"]))
                az.append(float(row["az"]))
            except Exception:
                # Skip malformed rows
                continue
    if not t_s:
        raise RuntimeError("No valid rows read from accelerometer CSV")
    return t_s, ax, ay, az


def plot_accl(t_s: List[float], ax: List[float], ay: List[float], az: List[float], title: str = "Accelerometer over Time"):
    plt.figure(figsize=(12, 6))
    plt.plot(t_s, ax, label="ax")
    plt.plot(t_s, ay, label="ay")
    plt.plot(t_s, az, label="az")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (parser units)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot accelerometer CSV (t_s, ax, ay, az) over time")
    parser.add_argument("csv", help="Path to accelerometer CSV produced by extract_accel.py")
    parser.add_argument("-o", "--output", help="Path to save the PNG (defaults next to CSV)")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        t_s, ax, ay, az = read_accl_csv(csv_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.abspath(args.output) if args.output else os.path.join(os.path.dirname(csv_path), f"{stem}_plot_accl.png")

    plot_accl(t_s, ax, ay, az, title=f"{stem} - Accelerometer")

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
