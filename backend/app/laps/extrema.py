"""Lap-vs-lap comparison: ported from legacy/overlay/lap_timer_v7.py:147-293.

The original rendered matplotlib PNGs from an interactive `input()` CLI
loop. Here the same math (local speed maxima/minima = braking/corner points,
straight-line speeds) is kept, but it returns plain data structures — the
webapp renders this as an interactive chart in the browser instead of a
static image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LapComparison:
    lap_a: int
    lap_b: int
    duration_a: float
    duration_b: float
    series_a: pd.DataFrame  # columns: rel_time, speed
    series_b: pd.DataFrame
    maxima_a: list[tuple[float, float]]  # (rel_time, speed)
    minima_a: list[tuple[float, float]]
    maxima_b: list[tuple[float, float]]
    minima_b: list[tuple[float, float]]


def find_local_extrema(values: pd.Series | list[float], window: int = 30, mode: str = "max") -> list[tuple[int, float]]:
    """Returns (index, value) for points that are the local max/min within
    +/- `window` samples on both sides — i.e. corner apexes (min) or braking
    points / straight-line peaks (max), not just noisy single-sample spikes.
    """
    vals = list(values)
    extrema = []
    for i in range(window, len(vals) - window):
        center = vals[i]
        neighborhood = vals[i - window:i] + vals[i + 1:i + window + 1]
        if mode == "max" and all(center >= x for x in neighborhood):
            extrema.append((i, center))
        elif mode == "min" and all(center <= x for x in neighborhood):
            extrema.append((i, center))
    return extrema


def _lap_slice(df: pd.DataFrame, lap_indices: list[int], lap_number: int) -> pd.DataFrame:
    s_idx, e_idx = lap_indices[lap_number - 1], lap_indices[lap_number]
    sl = df.iloc[s_idx:e_idx].copy()
    sl["rel_time"] = sl["time"] - sl.iloc[0]["time"]
    return sl


def compare_laps(
    df: pd.DataFrame,
    lap_table: pd.DataFrame,
    lap_indices: list[int],
    lap_a: int,
    lap_b: int,
    extrema_window: int = 30,
) -> LapComparison:
    n_laps = len(lap_indices) - 1
    if not (1 <= lap_a <= n_laps) or not (1 <= lap_b <= n_laps):
        raise ValueError(f"Lap numbers must be within 1..{n_laps}")

    slice_a = _lap_slice(df, lap_indices, lap_a)
    slice_b = _lap_slice(df, lap_indices, lap_b)

    max_a = find_local_extrema(slice_a["speed"], window=extrema_window, mode="max")
    min_a = find_local_extrema(slice_a["speed"], window=extrema_window, mode="min")
    max_b = find_local_extrema(slice_b["speed"], window=extrema_window, mode="max")
    min_b = find_local_extrema(slice_b["speed"], window=extrema_window, mode="min")

    t_a = slice_a["rel_time"].to_numpy()
    t_b = slice_b["rel_time"].to_numpy()

    def _to_points(extrema, t):
        return [(float(t[i]), float(v)) for i, v in extrema]

    dur_a = float(lap_table.loc[lap_table["lap"] == lap_a, "duration"].iloc[0])
    dur_b = float(lap_table.loc[lap_table["lap"] == lap_b, "duration"].iloc[0])

    return LapComparison(
        lap_a=lap_a,
        lap_b=lap_b,
        duration_a=dur_a,
        duration_b=dur_b,
        series_a=slice_a[["rel_time", "speed"]].reset_index(drop=True),
        series_b=slice_b[["rel_time", "speed"]].reset_index(drop=True),
        maxima_a=_to_points(max_a, t_a),
        minima_a=_to_points(min_a, t_a),
        maxima_b=_to_points(max_b, t_b),
        minima_b=_to_points(min_b, t_b),
    )
