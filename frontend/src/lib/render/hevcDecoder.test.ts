import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The real loader pulls in the libde265 wasm build; these tests only reach
// the gating decision in front of it, which deliberately returns before any
// of that is touched.
vi.mock('@yume-chan/libde265', () => ({ default: vi.fn() }))
vi.mock('@yume-chan/libde265/libde265.wasm?url', () => ({ default: '/libde265.wasm' }))

const registerDecoder = vi.fn()
vi.mock('mediabunny', () => ({
  registerDecoder: (...args: unknown[]) => registerDecoder(...args),
  // Registration takes a *class*, so this has to be extensible.
  CustomVideoDecoder: class {},
  VideoSample: class {},
}))

const { ensureSoftwareHevcDecoder } = await import('./hevcDecoder')

const hevcConfig: VideoDecoderConfig = { codec: 'hvc1.1.6.L150.90', codedWidth: 3840, codedHeight: 2160 }

/** Registration is a global, one-shot side effect, so tests that need to
 * observe it have to be the ones driving it -- hence checking call counts
 * rather than asserting a fresh registration each time. */
beforeEach(() => {
  registerDecoder.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ensureSoftwareHevcDecoder', () => {
  it('stays out of the way when the browser decodes this file natively', async () => {
    // The whole point of the gate: forcing software decode on a machine with
    // working hardware HEVC would be an order-of-magnitude slowdown for
    // nothing.
    vi.stubGlobal('VideoDecoder', { isConfigSupported: async () => ({ supported: true }) })

    await expect(ensureSoftwareHevcDecoder(hevcConfig)).resolves.toBe(false)
    expect(registerDecoder).not.toHaveBeenCalled()
  })

  it('registers the software decoder when the browser rejects the config', async () => {
    // Chromium ships no software HEVC decoder, so on a Linux box without
    // VAAPI (NVIDIA/AMD, VMs) this is the normal answer for HEVC footage.
    vi.stubGlobal('VideoDecoder', { isConfigSupported: async () => ({ supported: false }) })

    await expect(ensureSoftwareHevcDecoder(hevcConfig)).resolves.toBe(true)
    expect(registerDecoder).toHaveBeenCalledTimes(1)
  })

  it('treats a config the browser cannot even parse as undecodable', async () => {
    vi.stubGlobal('VideoDecoder', {
      isConfigSupported: async () => {
        throw new TypeError('unsupported config')
      },
    })

    await expect(ensureSoftwareHevcDecoder(hevcConfig)).resolves.toBe(true)
  })

  it('ignores non-HEVC tracks entirely', async () => {
    // H.264 gaps are real but this decoder doesn't address them, so claiming
    // the track would swap a clear error for a broken render.
    vi.stubGlobal('VideoDecoder', { isConfigSupported: async () => ({ supported: false }) })

    await expect(ensureSoftwareHevcDecoder({ codec: 'avc1.640028', codedWidth: 1920, codedHeight: 1080 })).resolves.toBe(false)
    expect(registerDecoder).not.toHaveBeenCalled()
  })

  it('accepts both spellings of the HEVC codec string', async () => {
    // 'hvc1' and 'hev1' differ only in where parameter sets live; both are
    // HEVC and both appear in action-cam MP4s.
    vi.stubGlobal('VideoDecoder', { isConfigSupported: async () => ({ supported: false }) })

    await expect(ensureSoftwareHevcDecoder({ codec: 'hev1.2.4.L153.B0', codedWidth: 3840, codedHeight: 2160 })).resolves.toBe(true)
  })

  it('does nothing when the track has no decoder config at all', async () => {
    await expect(ensureSoftwareHevcDecoder(null)).resolves.toBe(false)
    expect(registerDecoder).not.toHaveBeenCalled()
  })

  it('registers at most once across renders', async () => {
    vi.stubGlobal('VideoDecoder', { isConfigSupported: async () => ({ supported: false }) })
    // A fresh module instance, because the "have we registered yet" flag is
    // module state that earlier tests in this file have already tripped.
    vi.resetModules()
    registerDecoder.mockClear()
    const fresh = await import('./hevcDecoder')

    await fresh.ensureSoftwareHevcDecoder(hevcConfig)
    await fresh.ensureSoftwareHevcDecoder(hevcConfig)

    // Registration is global and permanent in mediabunny; doing it twice would
    // leave a duplicate in its decoder list forever.
    expect(registerDecoder).toHaveBeenCalledTimes(1)
  })
})
