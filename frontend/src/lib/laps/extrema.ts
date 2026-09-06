/**
 * Lap-vs-lap comparison: ported from backend/app/laps/extrema.py.
 */

import type { LapTableRow, TelemetryRow } from './detection'

// Field names mirror LapComparison's Python dataclass columns verbatim
// (like detection.ts's TelemetryRow/LapTableRow) rather than camelCase --
// this is data shape, not an API surface, and staying byte-for-byte
// comparable to extrema.py's fields keeps this port easy to audit.
export interface LapComparison {
  lap_a: number
  lap_b: number
  duration_a: number
  duration_b: number
  series_a: { rel_time: number; speed: number }[]
  series_b: { rel_time: number; speed: number }[]
  maxima_a: [number, number][]
  minima_a: [number, number][]
  maxima_b: [number, number][]
  minima_b: [number, number][]
}

/** Returns [index, value] for points that are the local max/min within +/-
 * `window` samples on both sides -- corner apexes (min) or braking points /
 * straight-line peaks (max), not just noisy single-sample spikes. */
export function findLocalExtrema(
  values: number[],
  window = 30,
  mode: 'max' | 'min' = 'max',
): [number, number][] {
  const extrema: [number, number][] = []
  for (let i = window; i < values.length - window; i++) {
    const center = values[i]
    const neighborhood = [...values.slice(i - window, i), ...values.slice(i + 1, i + window + 1)]
    if (mode === 'max' && neighborhood.every((x) => center >= x)) extrema.push([i, center])
    else if (mode === 'min' && neighborhood.every((x) => center <= x)) extrema.push([i, center])
  }
  return extrema
}

function lapSlice(rows: TelemetryRow[], lapIndices: number[], lapNumber: number): (TelemetryRow & { rel_time: number })[] {
  const sIdx = lapIndices[lapNumber - 1]
  const eIdx = lapIndices[lapNumber]
  const slice = rows.slice(sIdx, eIdx)
  const t0 = slice.length ? slice[0].time : 0
  return slice.map((r) => ({ ...r, rel_time: r.time - t0 }))
}

export function compareLaps(
  rows: TelemetryRow[],
  lapTable: LapTableRow[],
  lapIndices: number[],
  lapA: number,
  lapB: number,
  extremaWindow = 30,
): LapComparison {
  const nLaps = lapIndices.length - 1
  if (!(lapA >= 1 && lapA <= nLaps) || !(lapB >= 1 && lapB <= nLaps)) {
    throw new Error(`Lap numbers must be within 1..${nLaps}`)
  }

  const sliceA = lapSlice(rows, lapIndices, lapA)
  const sliceB = lapSlice(rows, lapIndices, lapB)

  const speedA = sliceA.map((r) => r.speed)
  const speedB = sliceB.map((r) => r.speed)
  const maxA = findLocalExtrema(speedA, extremaWindow, 'max')
  const minA = findLocalExtrema(speedA, extremaWindow, 'min')
  const maxB = findLocalExtrema(speedB, extremaWindow, 'max')
  const minB = findLocalExtrema(speedB, extremaWindow, 'min')

  const toPoints = (extrema: [number, number][], slice: (TelemetryRow & { rel_time: number })[]): [number, number][] =>
    extrema.map(([i, v]) => [slice[i].rel_time, v])

  const durationA = lapTable.find((l) => l.lap === lapA)!.duration
  const durationB = lapTable.find((l) => l.lap === lapB)!.duration

  return {
    lap_a: lapA,
    lap_b: lapB,
    duration_a: durationA,
    duration_b: durationB,
    series_a: sliceA.map((r) => ({ rel_time: r.rel_time, speed: r.speed })),
    series_b: sliceB.map((r) => ({ rel_time: r.rel_time, speed: r.speed })),
    maxima_a: toPoints(maxA, sliceA),
    minima_a: toPoints(minA, sliceA),
    maxima_b: toPoints(maxB, sliceB),
    minima_b: toPoints(minB, sliceB),
  }
}
