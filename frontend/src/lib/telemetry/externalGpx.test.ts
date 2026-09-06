import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { computeSpeedKmh, extractExternalGpx, loadGpxPoints } from './externalGpx'

// Ported from backend/tests/test_external_gpx.py (since removed along with
// the rest of the server-side telemetry pipeline -- see backend/readme.md).
// This fixture used to live in backend/tests/fixtures/ and was moved here
// when this became the only test suite still using it.
const sampleGpx = readFileSync(resolve(import.meta.dirname, 'fixtures/Krakow - Zarki.gpx'), 'utf-8')

describe('loadGpxPoints', () => {
  it('parses the real fixture', () => {
    const points = loadGpxPoints(sampleGpx)
    expect(points.length).toBeGreaterThan(0)
    for (let i = 1; i < points.length; i++) {
      expect(points[i].timestamp.getTime()).toBeGreaterThanOrEqual(points[i - 1].timestamp.getTime())
    }
    // sample route is in southern Poland
    for (const p of points) {
      expect(p.lat).toBeGreaterThanOrEqual(49.0)
      expect(p.lat).toBeLessThanOrEqual(51.0)
      expect(p.lon).toBeGreaterThanOrEqual(18.0)
      expect(p.lon).toBeLessThanOrEqual(21.0)
    }
  })
})

describe('computeSpeedKmh', () => {
  it('is always non-negative', () => {
    const points = loadGpxPoints(sampleGpx)
    const speed = computeSpeedKmh(points)
    expect(speed.every((v) => v >= 0)).toBe(true)
    // no prior point to derive speed from at index 0 -- the median filter
    // blends that 0 sentinel with its neighbor, so it's small, not exactly 0
    expect(speed[0]).toBeLessThan(Math.max(...speed))
  })
})

describe('extractExternalGpx', () => {
  it('aligns to an explicit video start time', () => {
    const points = loadGpxPoints(sampleGpx)
    const firstPointTime = points[0].timestamp
    const videoStartTime = new Date(firstPointTime.getTime() - 10_000)

    const result = extractExternalGpx(sampleGpx, 120.0, { targetFps: 1.0, videoStartTime })

    expect(result.sourceName).toBe('external_gpx')
    expect(result.hasAccel).toBe(false)
    expect(result.rows.length).toBeGreaterThan(0)
    expect(Math.min(...result.rows.map((r) => r.time))).toBeGreaterThanOrEqual(0)
    expect(Math.max(...result.rows.map((r) => r.time))).toBeLessThanOrEqual(120.0)
  })

  it('is empty when the window excludes every point', () => {
    const points = loadGpxPoints(sampleGpx)
    const videoStartTime = new Date(points[0].timestamp.getTime() + 24 * 60 * 60 * 1000)

    const result = extractExternalGpx(sampleGpx, 60.0, { targetFps: 1.0, videoStartTime })
    expect(result.rows).toEqual([])
  })
})
