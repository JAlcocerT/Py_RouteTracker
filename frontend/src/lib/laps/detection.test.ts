import { describe, expect, it } from 'vitest'
import {
  _estimateHeading,
  _resolveCrossing,
  detectLaps,
  getCoordinatesAtTime,
  haversineDistanceM,
  type TelemetryRow,
} from './detection'

// Ported from backend/tests/test_laps_detection.py.
const DEG_PER_METER_AT_EQUATOR = 1 / 111_000

function circularTrackDf(nLaps: number, lapTimeS: number, dt: number, trackRadiusM = 100.0): TelemetryRow[] {
  const radiusDeg = trackRadiusM * DEG_PER_METER_AT_EQUATOR
  const n = Math.round((nLaps * lapTimeS) / dt)
  const t = Array.from({ length: n }, (_, i) => i * dt)
  const circumferenceM = 2 * Math.PI * trackRadiusM
  const speedKmh = (circumferenceM / lapTimeS) * 3.6
  return t.map((time) => {
    const theta = (2 * Math.PI * (time % lapTimeS)) / lapTimeS
    return {
      time,
      lat: radiusDeg * Math.cos(theta),
      lon: radiusDeg * Math.sin(theta),
      speed: speedKmh,
      lat_g: 0,
      lon_g: 0,
    }
  })
}

describe('haversineDistanceM', () => {
  it('matches the known 1-degree-of-latitude distance', () => {
    const d = haversineDistanceM(0.0, 0.0, 1.0, 0.0) as number
    expect(d).toBeCloseTo(111_195, -2)
  })

  it('vectorized matches scalar', () => {
    const lat1 = [0.0, 10.0]
    const lon1 = [0.0, 20.0]
    const scalar = lat1.map((a, i) => haversineDistanceM(a, lon1[i], 0.0, 0.0) as number)
    const vector = haversineDistanceM(lat1, lon1, 0.0, 0.0) as number[]
    vector.forEach((v, i) => expect(v).toBeCloseTo(scalar[i], 5))
  })
})

describe('getCoordinatesAtTime', () => {
  it('finds the nearest sample', () => {
    const df = circularTrackDf(1, 60, 0.5)
    const [lat, lon] = getCoordinatesAtTime(df, 0.0)
    expect(lat).toBeCloseTo(100 * DEG_PER_METER_AT_EQUATOR)
    expect(lon).toBeCloseTo(0.0, 9)
  })
})

describe('detectLaps', () => {
  it('finds the expected lap count', () => {
    const df = circularTrackDf(4, 60, 0.5, 100.0)
    const [startLat, startLon] = getCoordinatesAtTime(df, 0.0)
    const result = detectLaps(df, startLat, startLon, 15.0, 30.0)
    expect(result.lapIndices.length).toBe(4)
    expect(result.lapTable.length).toBe(3)
    for (const row of result.lapTable) expect(Math.abs(row.duration - 60.0)).toBeLessThanOrEqual(1.0)
  })

  it('annotates lap numbers monotonically', () => {
    const df = circularTrackDf(3, 40, 0.5, 80.0)
    const [startLat, startLon] = getCoordinatesAtTime(df, 0.0)
    const result = detectLaps(df, startLat, startLon, 15.0, 20.0)
    const laps = result.annotatedDf.map((r) => r.lap)
    for (let i = 1; i < laps.length; i++) expect(laps[i]).toBeGreaterThanOrEqual(laps[i - 1])
    expect(Math.max(...laps)).toBe(result.lapIndices.length + 1)
  })

  it('returns an empty result for an empty input', () => {
    const result = detectLaps([], 0.0, 0.0)
    expect(result.lapIndices).toEqual([])
    expect(result.lapTable).toEqual([])
  })

  it('resets lap_elapsed_s at each crossing and grows within a lap', () => {
    const df = circularTrackDf(3, 60, 0.5, 100.0)
    const [startLat, startLon] = getCoordinatesAtTime(df, 0.0)
    const result = detectLaps(df, startLat, startLon, 15.0, 30.0)
    expect(result.lapIndices.length).toBeGreaterThanOrEqual(2)
    const firstIdx = result.lapIndices[0]
    expect(result.annotatedDf[firstIdx].lap_elapsed_s).toBeLessThan(1.0)
    const secondIdx = result.lapIndices[1]
    expect(result.annotatedDf[secondIdx - 1].lap_elapsed_s).toBeGreaterThan(55.0)
  })

  it('respects min_lap_time and ignores noise near the line', () => {
    const n = 200
    // deterministic tiny jitter instead of Python's seeded RNG -- the
    // assertion only needs "small noise doesn't leave the zone", not the
    // exact same random sequence as the original test.
    const df: TelemetryRow[] = Array.from({ length: n }, (_, i) => ({
      time: i * 0.5,
      lat: ((i % 7) - 3) * 1e-8,
      lon: 0,
      speed: 0,
      lat_g: 0,
      lon_g: 0,
    }))
    const result = detectLaps(df, 0.0, 0.0, 15.0, 30.0)
    expect(result.lapIndices.length).toBeLessThanOrEqual(1)
  })
})

describe('_estimateHeading', () => {
  it('matches straight-line east travel', () => {
    const eastM = [-10.0, -5.0, 0.0, 5.0, 10.0]
    const northM = [0, 0, 0, 0, 0]
    const heading = _estimateHeading(eastM, northM, 2, 2)
    expect(heading).not.toBeNull()
    expect(heading![0]).toBeCloseTo(1.0, 9)
    expect(heading![1]).toBeCloseTo(0.0, 9)
  })

  it('is null when stationary', () => {
    expect(_estimateHeading([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 2, 2)).toBeNull()
  })
})

describe('_resolveCrossing', () => {
  it('interpolates between straddling samples', () => {
    const t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    const eastM = [-4.0, -3.0, -2.0, -1.0, -0.3, 0.7, 1.7]
    const northM = new Array(7).fill(0)
    const [idx, crossT] = _resolveCrossing(eastM, northM, t, 2, 6, 4, [1.0, 0.0])
    expect(crossT).toBeCloseTo(2.15, 9)
    expect(idx).toBe(4)
  })

  it('falls back to the nearest sample without a heading', () => {
    const t = [0.0, 1.0, 2.0]
    const eastM = [1.0, 0.1, 1.0]
    const northM = [0, 0, 0]
    const [idx, crossT] = _resolveCrossing(eastM, northM, t, 0, 2, 1, null)
    expect(idx).toBe(1)
    expect(crossT).toBeCloseTo(1.0)
  })
})
