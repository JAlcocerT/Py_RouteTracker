import { describe, expect, it } from 'vitest'
import { resampleToGrid, smoothSpeedOutliers } from './resample'

// Ported from backend/tests/test_speed_smoothing.py -- see that file for the
// real-world bug report (a go-kart session showing a 300 km/h lap max) this
// guards against.
describe('smoothSpeedOutliers', () => {
  it('rejects an isolated spike', () => {
    const speed = [84.0, 85.0, 86.0, 300.0, 85.0, 84.0, 85.0]
    const cleaned = smoothSpeedOutliers(speed)
    expect(Math.max(...cleaned)).toBeLessThan(100)
    // genuine neighboring values are left essentially untouched (matches
    // the Python test's `pytest.approx(84.5, abs=1.0)`)
    expect(Math.abs(cleaned[0] - 84.5)).toBeLessThanOrEqual(1.0)
  })

  it('preserves a sustained change', () => {
    const speed = [20.0, 20.0, 60.0, 65.0, 70.0, 70.0, 70.0]
    const cleaned = smoothSpeedOutliers(speed)
    expect(cleaned[cleaned.length - 1]).toBeCloseTo(70.0, 5)
    expect(cleaned[3]).toBeGreaterThan(50)
  })

  it('is a no-op on a short series', () => {
    const speed = [10.0, 20.0]
    expect(smoothSpeedOutliers(speed)).toEqual([10.0, 20.0])
  })

  it('widens the window to match the sample rate', () => {
    const n = 30
    const baseline = 85.0
    const speed = new Array(n).fill(baseline)
    const burstStart = 13
    const burstLen = 4
    for (let i = burstStart; i < burstStart + burstLen; i++) speed[i] = 300.0
    const time = Array.from({ length: n }, (_, i) => i * 0.05) // 20 Hz

    const withoutTime = smoothSpeedOutliers(speed.slice())
    expect(Math.max(...withoutTime)).toBeGreaterThan(150)

    const withTime = smoothSpeedOutliers(speed.slice(), 5, 5.0, time)
    expect(Math.max(...withTime)).toBeLessThan(100)
  })

  it('rejects a two-sample burst', () => {
    const speed = [84.0, 85.0, 86.0, 300.0, 305.0, 85.0, 84.0, 85.0]
    const cleaned = smoothSpeedOutliers(speed)
    expect(Math.max(...cleaned)).toBeLessThan(100)
  })
})

describe('resampleToGrid', () => {
  it('linearly interpolates between known samples and clamps at the edges', () => {
    const rows = [
      { time: 1, v: 10 },
      { time: 3, v: 30 },
    ]
    const grid = resampleToGrid(rows, 4, 1, ['v'])
    // grid times: 0, 1, 2, 3
    expect(grid.map((r) => r.time)).toEqual([0, 1, 2, 3])
    expect(grid[0].v).toBeCloseTo(10) // before first sample -> clamped
    expect(grid[1].v).toBeCloseTo(10)
    expect(grid[2].v).toBeCloseTo(20) // halfway between t=1 (10) and t=3 (30)
    expect(grid[3].v).toBeCloseTo(30)
  })

  it('returns NaN-filled rows for an empty input', () => {
    const grid = resampleToGrid([] as { time: number; v: number }[], 2, 1, ['v'])
    expect(grid.length).toBe(2)
    expect(Number.isNaN(grid[0].v)).toBe(true)
  })
})
