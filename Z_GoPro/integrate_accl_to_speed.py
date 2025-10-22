#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt  # optional; only used if --show
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


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
                continue

    if not t_s:
        raise RuntimeError("No valid rows read from accelerometer CSV")

    # sort by time if not strictly increasing
    if any(t2 < t1 for t1, t2 in zip(t_s, t_s[1:])):
        idx = sorted(range(len(t_s)), key=lambda i: t_s[i])
        t_s = [t_s[i] for i in idx]
        ax = [ax[i] for i in idx]
        ay = [ay[i] for i in idx]
        az = [az[i] for i in idx]

    return t_s, ax, ay, az


def integrate_trapezoid(t: List[float], a: List[float]) -> List[float]:
    n = len(t)
    v = [0.0] * n
    if n == 0:
        return v
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt < 0:
            dt = 0.0
        v[i] = v[i - 1] + 0.5 * (a[i - 1] + a[i]) * dt
    return v


def write_speed_csv(out_path: str, t: List[float], vx: List[float], vy: List[float], vz: List[float]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "vx", "vy", "vz"])
        for i in range(len(t)):
            writer.writerow([t[i], vx[i], vy[i], vz[i]])


def plot_speed(t: List[float], vx: List[float], vy: List[float], vz: List[float], title: str) -> None:
    if not _HAS_MPL:
        print("WARNING: matplotlib not installed; cannot show plot. Install with: pip install matplotlib", file=sys.stderr)
        return
    plt.figure(figsize=(12, 6))
    plt.plot(t, vx, label="vx")
    plt.plot(t, vy, label="vy")
    plt.plot(t, vz, label="vz")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (accel_units*s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Integrate accelerometer CSV (t_s, ax, ay, az) to speed assuming initial speed 0")
    parser.add_argument("csv", help="Path to accelerometer CSV produced by extract_accel.py")
    parser.add_argument("-o", "--output", help="Path to save the speed CSV (defaults next to input)")
    parser.add_argument("--show", action="store_true", help="Show a plot of vx, vy, vz over time")
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

    vx = integrate_trapezoid(t_s, ax)
    vy = integrate_trapezoid(t_s, ay)
    vz = integrate_trapezoid(t_s, az)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.abspath(args.output) if args.output else os.path.join(os.path.dirname(csv_path), f"{stem}_speed.csv")

    try:
        write_speed_csv(out_path, t_s, vx, vy, vz)
        print(f"Saved speed CSV: {out_path}")
    except Exception as e:
        print(f"ERROR writing CSV: {e}", file=sys.stderr)
        sys.exit(3)

    if args.show:
        title = f"{stem} - Speed (from accel)"
        plot_speed(t_s, vx, vy, vz, title)


if __name__ == "__main__":
    main()
