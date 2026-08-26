import { describe, expect, it } from 'vitest'
import type { LapTableRow, TelemetryRow } from './detection'
import { compareLaps, findLocalExtrema } from './extrema'

// Ported from backend/tests/test_extrema.py.
describe('findLocalExtrema', () => {
  it('finds a single peak', () => {
    const values = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    expect(findLocalExtrema(values, 3, 'max')).toEqual([[5, 5]])
  })

  it('finds a single trough', () => {
    const values = [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    expect(findLocalExtrema(values, 3, 'min')).toEqual([[5, 0]])
  })

  it('is empty when the window is wider than the series', () => {
    expect(findLocalExtrema([1, 2, 3], 5, 'max')).toEqual([])
  })
})

function twoLapFixture(): { df: TelemetryRow[]; lapTable: LapTableRow[]; lapIndices: number[] } {
  const t: number[] = []
  for (let x = 0; x < 40; x += 0.5) t.push(x)
  const lap1Speed = t.map((time) => 100 + 20 * Math.sin((2 * Math.PI * time) / 40))
  const lap2Speed = t.map((time) => 100 + 30 * Math.sin((2 * Math.PI * time) / 40))

  const df: TelemetryRow[] = [
    ...t.map((time, i) => ({ time, lat: 0, lon: 0, speed: lap1Speed[i], lat_g: 0, lon_g: 0 })),
    ...t.map((time, i) => ({ time: time + 40, lat: 0, lon: 0, speed: lap2Speed[i], lat_g: 0, lon_g: 0 })),
  ]
  const lapIndices = [0, t.length, t.length * 2 - 1]
  const lapTable: LapTableRow[] = [
    { lap: 1, start_time: 0.0, end_time: 40.0, duration: 40.0, avg_speed: 100.0, max_speed: 120.0 },
    { lap: 2, start_time: 40.0, end_time: 79.5, duration: 39.5, avg_speed: 100.0, max_speed: 130.0 },
  ]
  return { df, lapTable, lapIndices }
}

describe('compareLaps', () => {
  it('returns the expected shape', () => {
    const { df, lapTable, lapIndices } = twoLapFixture()
    const comparison = compareLaps(df, lapTable, lapIndices, 1, 2, 5)

    expect(comparison.lap_a).toBe(1)
    expect(comparison.lap_b).toBe(2)
    expect(comparison.duration_a).toBeCloseTo(40.0)
    expect(comparison.duration_b).toBeCloseTo(39.5)
    expect(comparison.series_a.length).toBeGreaterThan(0)
    expect(comparison.series_b.length).toBeGreaterThan(0)
    expect(comparison.maxima_b.length).toBeGreaterThanOrEqual(1)
    const maxA = Math.max(...comparison.maxima_a.map(([, v]) => v))
    const maxB = Math.max(...comparison.maxima_b.map(([, v]) => v))
    expect(maxB).toBeGreaterThan(maxA)
  })

  it('rejects out-of-range lap numbers', () => {
    const { df, lapTable, lapIndices } = twoLapFixture()
    expect(() => compareLaps(df, lapTable, lapIndices, 1, 5)).toThrow()
  })
})
