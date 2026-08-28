import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The real loader pulls in the libde265 wasm build; these tests only reach
// the gating decision in front of it, which deliberately returns before any
// of that is touched.
vi.mock('@yume-chan/libde265', () => ({ default: vi.fn() }))
vi.mock('@yume-chan/libde265/libde265.wasm?url', () => ({ default: '/libde265.wasm' }))

const registerDecoder = vi.fn()
/** Captures what the decoder asks `VideoSample` to be built with, which is
 * where the frame's timestamp, duration and pixel layout are decided. */
const videoSampleInits: { data: Uint8Array; init: Record<string, unknown> }[] = []
vi.mock('mediabunny', () => ({
  registerDecoder: (...args: unknown[]) => registerDecoder(...args),
  // Registration takes a *class*, so this has to be extensible.
  CustomVideoDecoder: class {},
  VideoSample: class {
    timestamp: number
    duration: number
    constructor(data: Uint8Array, init: Record<string, unknown>) {
      videoSampleInits.push({ data, init })
      this.timestamp = init.timestamp as number
      this.duration = init.duration as number
    }
    close() {}
  },
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

/**
 * A stand-in for libde265's decoder that returns one 4:2:0 image per frame
 * pushed, echoing back the timestamp it was given -- enough to exercise how
 * frames are turned into VideoSamples without running the real wasm.
 */
function fakeLibde265(options: { bitsPerSample?: number; planeByteLength?: (needed: number) => number } = {}) {
  const bitsPerSample = options.bitsPerSample ?? 8
  const bytesPerSample = bitsPerSample > 8 ? 2 : 1
  const width = 4
  const height = 2
  const queue: bigint[] = []

  const makeImage = (pts: bigint) => ({
    chromaFormat: 1,
    isFullRange: false,
    colorPrimaries: 1,
    transferCharacteristics: 1,
    matrixCoefficients: 1,
    pts,
    getWidth: (c: number) => (c === 0 ? width : width / 2),
    getHeight: (c: number) => (c === 0 ? height : height / 2),
    getBitsPerPixel: () => bitsPerSample,
    getImagePlane: (c: number) => {
      const w = c === 0 ? width : width / 2
      const h = c === 0 ? height : height / 2
      const stride = w * bytesPerSample
      const needed = stride * h
      // The real binding hands back a view sized from the bit depth, which is
      // too short for high bit depth; `planeByteLength` reproduces that.
      const exposed = options.planeByteLength ? options.planeByteLength(needed) : needed
      const backing = new Uint8Array(needed)
      backing.fill(c === 0 ? 0x40 : 0x80)
      return { width: w, height: h, stride, bytes: backing.subarray(0, exposed) }
    },
    delete: () => {},
  })

  const decoder = {
    pushNal: () => 0,
    pushEndOfFrame: () => {},
    pushEndOfNal: () => {},
    flushData: () => 0,
    reset: () => {
      queue.length = 0
    },
    delete: () => {},
    decode: () => ({ error: 13, more: false }), // ERROR_WAITING_FOR_INPUT_DATA
    getNextPicture: () => (queue.length > 0 ? makeImage(queue.shift()!) : null),
    enqueue: (pts: bigint) => queue.push(pts),
  }

  return {
    module: {
      Decoder: function () {
        return decoder
      } as unknown as new () => typeof decoder,
      Error: { ERROR_WAITING_FOR_INPUT_DATA: 13 },
      isOk: (e: number) => e === 0,
      getErrorText: (e: number) => `error ${e}`,
    },
    decoder,
  }
}

/** hvcC with one SPS array whose NAL is deliberately unparseable, so the
 * decoder falls back to "no reordering" -- keeping these tests focused on
 * frame construction rather than reorder depth. */
const HVCC = Uint8Array.from([...Array(21).fill(0), 3, 1, 0x21, 0, 1, 0, 2, 0, 0])

async function decodeOne(fake: ReturnType<typeof fakeLibde265>, packet: { timestamp: number; duration: number }) {
  const initLibde265 = (await import('@yume-chan/libde265')).default as unknown as ReturnType<typeof vi.fn>
  initLibde265.mockResolvedValue(fake.module)
  const { WasmHevcDecoder } = await import('./hevcDecoder')

  const emitted: { timestamp: number; duration: number }[] = []
  const decoder = new (WasmHevcDecoder as unknown as new () => InstanceType<typeof WasmHevcDecoder>)()
  Object.assign(decoder, {
    config: { codec: 'hvc1.1.6.L120.90', description: HVCC, codedWidth: 4, codedHeight: 2 },
    onSample: (s: { timestamp: number; duration: number }) => emitted.push(s),
  })
  await decoder.init()
  fake.decoder.enqueue(BigInt(Math.round(packet.timestamp * 1_000_000)))
  await decoder.decode({ data: new Uint8Array([0, 0, 0, 1, 0x26]), timestamp: packet.timestamp, duration: packet.duration, type: 'key' } as never)
  return emitted
}

describe('WasmHevcDecoder frame construction', () => {
  beforeEach(() => {
    videoSampleInits.length = 0
    vi.resetModules()
  })

  it('carries the packet duration onto the decoded frame', async () => {
    // The regression that made rendered files unplayable: libde265 knows
    // nothing about durations, so without carrying them across every sample
    // came out with duration 0 and the output video collapsed to a single
    // frame -- audio played, the picture never moved.
    const emitted = await decodeOne(fakeLibde265(), { timestamp: 0.5, duration: 1 / 30 })

    expect(emitted).toHaveLength(1)
    expect(emitted[0].duration).toBeCloseTo(1 / 30, 6)
    expect(emitted[0].timestamp).toBeCloseTo(0.5, 6)
  })

  it('never emits a zero duration, even if a frame timestamp goes unmatched', async () => {
    const fake = fakeLibde265()
    const initLibde265 = (await import('@yume-chan/libde265')).default as unknown as ReturnType<typeof vi.fn>
    initLibde265.mockResolvedValue(fake.module)
    const { WasmHevcDecoder } = await import('./hevcDecoder')

    const emitted: { duration: number }[] = []
    const decoder = new (WasmHevcDecoder as unknown as new () => InstanceType<typeof WasmHevcDecoder>)()
    Object.assign(decoder, {
      config: { codec: 'hvc1.1.6.L120.90', description: HVCC, codedWidth: 4, codedHeight: 2 },
      onSample: (s: { duration: number }) => emitted.push(s),
    })
    await decoder.init()

    fake.decoder.enqueue(1_000_000n) // matches the packet below
    await decoder.decode({ data: new Uint8Array([0, 0, 0, 1, 0x26]), timestamp: 1, duration: 1 / 25, type: 'key' } as never)
    fake.decoder.enqueue(9_999_999n) // a timestamp no packet ever declared
    await decoder.decode({ data: new Uint8Array([0, 0, 0, 1, 0x26]), timestamp: 2, duration: 1 / 25, type: 'key' } as never)

    expect(emitted).toHaveLength(2)
    // Falls back to the last known duration rather than the zero that broke playback.
    expect(emitted[1].duration).toBeCloseTo(1 / 25, 6)
  })

  it('reads the whole plane when the binding under-reports its length', async () => {
    // libde265's binding sizes 10-bit plane views as width*height*bits/8,
    // which covers only two thirds of the picture; copying from that view
    // truncated every 10-bit frame. The full plane is allocated, so it has to
    // be re-derived from the same buffer.
    const emitted = await decodeOne(
      fakeLibde265({ bitsPerSample: 10, planeByteLength: (needed) => Math.floor((needed * 5) / 8) }),
      { timestamp: 0, duration: 1 / 30 },
    )

    expect(emitted).toHaveLength(1)
    // 4x2 luma + two 2x1 chroma planes, packed tight and 8 bits deep.
    const { data, init } = videoSampleInits[0]
    expect(init.format).toBe('I420')
    expect(data).toHaveLength(4 * 2 + 2 * 1 + 2 * 1)
    // Every sample was written; a truncated read would leave trailing zeroes.
    expect(Array.from(data).every((v) => v > 0)).toBe(true)
  })

  it('hands over 8-bit frames as tightly packed I420', async () => {
    await decodeOne(fakeLibde265(), { timestamp: 0, duration: 1 / 30 })

    const { data, init } = videoSampleInits[0]
    expect(init.format).toBe('I420')
    // Layout strides equal the plane widths -- no padding carried through.
    expect(init.layout).toEqual([
      { offset: 0, stride: 4 },
      { offset: 8, stride: 2 },
      { offset: 10, stride: 2 },
    ])
    expect(data).toHaveLength(12)
  })
})
