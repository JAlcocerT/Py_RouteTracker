import { describe, expect, it, vi } from 'vitest'

// gpmf-extract/gopro-telemetry need a real GoPro MP4 to do anything useful --
// no such fixture exists in this repo, so this test mocks both at the
// boundary and only exercises this module's own mapping/unit-conversion
// logic (GPS5 array layout -> {lat, lon, speed}, ACCL axis selection +
// median-magnitude normalization, resampling onto the target grid). See
// goproGpmf.ts's module docstring: real-hardware validation is still needed.
vi.mock('gpmf-extract', () => ({ default: vi.fn(async () => ({ rawData: new Uint8Array(), timing: {} })) }))

const gpsSamples = [
  { cts: 0, value: [50.0, 19.0, 200, 10.0, 10.1] }, // 10 m/s -> 36 km/h
  { cts: 500, value: [50.0001, 19.0001, 200, 12.0, 12.1] },
  { cts: 1000, value: [50.0002, 19.0002, 200, 14.0, 14.1] },
  { cts: 1500, value: [50.0003, 19.0003, 200, 16.0, 16.1] },
]

const accelSamples = [
  { cts: 0, value: [1000, 100, 50] },
  { cts: 250, value: [1000, 110, 60] },
  { cts: 500, value: [1000, 90, 40] },
  { cts: 750, value: [1000, 105, 55] },
  { cts: 1000, value: [1000, 95, 45] },
]

vi.mock('gopro-telemetry', () => ({
  default: vi.fn(async () => ({
    1: {
      streams: {
        GPS5: { samples: gpsSamples },
        ACCL: { samples: accelSamples },
      },
    },
  })),
}))

describe('extractGoProGpmf', () => {
  it('maps GPS5 samples to km/h and resamples onto the target grid', async () => {
    const { extractGoProGpmf } = await import('./goproGpmf')
    const progress: number[] = []
    const result = await extractGoProGpmf({} as File, 2.0, { targetFps: 2.0, onProgress: (p) => progress.push(p) }) // 2 fps -> grid at t=0,0.5,1,1.5

    expect(result.sourceName).toBe('gopro_embedded')
    expect(result.hasAccel).toBe(true)
    expect(result.rows.map((r) => r.time)).toEqual([0, 0.5, 1, 1.5])
    // first sample: speed2D=10 m/s -> 36 km/h
    expect(result.rows[0].speed).toBeCloseTo(36.0, 0)
    expect(result.rows[0].lat).toBeCloseTo(50.0, 4)
    expect(progress[progress.length - 1]).toBe(1)
  })

  it('resolves to an empty result instead of hanging when the video has no gpmd track', async () => {
    vi.resetModules()
    // Mirrors gpmf-extract's real behaviour for a video with no metadata
    // track: rejects with the plain string 'Track not found' rather than an
    // Error. Before the useWorker: false fix, the equivalent worker-based
    // path never settled the promise at all.
    vi.doMock('gpmf-extract', () => ({ default: vi.fn(async () => Promise.reject('Track not found')) }))
    vi.doMock('gopro-telemetry', () => ({ default: vi.fn() }))
    const { extractGoProGpmf } = await import('./goproGpmf')
    const result = await extractGoProGpmf({} as File, 2.0)
    expect(result.rows).toEqual([])
    expect(result.sourceName).toBe('gopro_embedded')
  })

  it('is empty when every GPS sample is the (0,0) no-fix sentinel', async () => {
    vi.resetModules()
    vi.doMock('gpmf-extract', () => ({ default: vi.fn(async () => ({ rawData: new Uint8Array(), timing: {} })) }))
    vi.doMock('gopro-telemetry', () => ({
      default: vi.fn(async () => ({
        1: { streams: { GPS5: { samples: [{ cts: 0, value: [0, 0, 0, 0, 0] }] } } },
      })),
    }))
    const { extractGoProGpmf } = await import('./goproGpmf')
    const result = await extractGoProGpmf({} as File, 2.0, { targetFps: 2.0 })
    expect(result.rows).toEqual([])
    expect(result.hasAccel).toBe(false)
  })
})
