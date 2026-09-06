import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderVideo } from './pipeline'
import { DEFAULT_RENDER_CONFIG } from './renderConfig'

// Mediabunny's Conversion talks to real codecs/WebCodecs, which don't exist
// in this test environment -- mocked at the module boundary so this only
// exercises renderVideo's own isValid/discardedTracks handling and how it
// engages the software HEVC decoder, not the codecs themselves.
interface FakeConversion {
  isValid: boolean
  discardedTracks: { track: { type: string; number: number; codec: string }; reason: string }[]
  execute: () => Promise<void>
}

const decoderConfig: VideoDecoderConfig = { codec: 'hvc1.1.6.L150.90', codedWidth: 3840, codedHeight: 2160 }
const primaryVideoTrack = {
  getDisplayWidth: vi.fn(async () => 640),
  getDisplayHeight: vi.fn(async () => 360),
  getDecoderConfig: vi.fn(async () => decoderConfig),
}
const executeMock = vi.fn(async () => undefined)
const initMock = vi.fn()
let conversionResult: FakeConversion

vi.mock('mediabunny', () => ({
  ALL_FORMATS: [],
  BlobSource: vi.fn(),
  BufferTarget: vi.fn(function (this: { buffer: ArrayBuffer }) {
    this.buffer = new ArrayBuffer(0)
  }),
  StreamTarget: vi.fn(),
  Mp4OutputFormat: vi.fn(),
  WebMOutputFormat: vi.fn(),
  Output: vi.fn(),
  Input: vi.fn(function (this: { getPrimaryVideoTrack: () => Promise<typeof primaryVideoTrack> }) {
    this.getPrimaryVideoTrack = async () => primaryVideoTrack
  }),
  Conversion: {
    init: (...args: unknown[]) => {
      initMock(...args)
      return Promise.resolve(conversionResult)
    },
  },
}))

// The real module loads a ~2.3MB wasm build of libavcodec, which has no place
// in a unit test. What matters here is only *whether* renderVideo consults it
// and what it does with the answer -- hevcDecoder.test.ts covers the gating
// decision itself.
const ensureSoftwareHevcDecoderMock = vi.fn(async (_config: VideoDecoderConfig | null) => false)
vi.mock('./hevcDecoder', () => ({
  ensureSoftwareHevcDecoder: (config: VideoDecoderConfig | null) => ensureSoftwareHevcDecoderMock(config),
}))

/** Records what the HUD is asked to draw, and with which options. */
const hudDraws: { time: number; clear: boolean | undefined }[] = []
vi.mock('./hudRenderer', () => ({
  HudRenderer: class {
    drawFrameAtTime(_ctx: unknown, time: number, options?: { clear?: boolean }) {
      hudDraws.push({ time, clear: options?.clear })
    }
  },
}))

class FakeOffscreenCanvas {
  width: number
  height: number
  constructor(width: number, height: number) {
    this.width = width
    this.height = height
  }
  getContext() {
    return {}
  }
}
vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas)

const row = { time: 0, lat: 0, lon: 0, speed: 0, lat_g: 0, lon_g: 0, lap: 0, last_lap_s: 0, lap_elapsed_s: 0 }
const baseOptions = { trimStart: 0, trimEnd: 1, config: DEFAULT_RENDER_CONFIG, annotatedRows: [row] }

const invalidWith = (discardedTracks: FakeConversion['discardedTracks']): FakeConversion => ({
  isValid: false,
  discardedTracks,
  execute: executeMock,
})
const valid = (): FakeConversion => ({ isValid: true, discardedTracks: [], execute: executeMock })

beforeEach(() => {
  hudDraws.length = 0
  executeMock.mockClear()
  initMock.mockClear()
  ensureSoftwareHevcDecoderMock.mockClear()
  ensureSoftwareHevcDecoderMock.mockImplementation(async () => false)
  conversionResult = valid()
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  // Present by default (WebCodecs itself works) -- the test that needs the
  // API genuinely missing overrides this for itself.
  vi.stubGlobal('VideoDecoder', class {})
  vi.stubGlobal('AudioDecoder', class {})
})

describe('renderVideo', () => {
  it('proceeds to execute when the conversion is valid', async () => {
    const result = await renderVideo({} as File, baseOptions)

    expect(executeMock).toHaveBeenCalledTimes(1)
    expect(result).toBeInstanceOf(Blob)
  })

  it('offers the software HEVC decoder the real track config, not a guessed one', async () => {
    // HEVC support is routinely partial -- a decoder that takes 8-bit Main at
    // 1080p may still refuse Main10 or 4K -- so probing with anything other
    // than the file's own config would answer the wrong question.
    await renderVideo({} as File, baseOptions)

    expect(ensureSoftwareHevcDecoderMock).toHaveBeenCalledWith(decoderConfig)
  })

  it('reports the software-decoding stage only when that decoder is actually engaged', async () => {
    const onStage = vi.fn()
    await renderVideo({} as File, { ...baseOptions, onStage })
    expect(onStage).not.toHaveBeenCalled()

    ensureSoftwareHevcDecoderMock.mockImplementation(async () => true)
    await renderVideo({} as File, { ...baseOptions, onStage })
    expect(onStage).toHaveBeenCalledWith('software-decoding')
  })

  it('renders in a single pass -- no transcode-and-retry', async () => {
    // The software decoder slots into Conversion's own decode step, so unlike
    // the whole-file transcode this replaced, the source is never converted to
    // an intermediate and re-read.
    ensureSoftwareHevcDecoderMock.mockImplementation(async () => true)

    await renderVideo({} as File, baseOptions)

    expect(initMock).toHaveBeenCalledTimes(1)
    expect(executeMock).toHaveBeenCalledTimes(1)
  })

  it('surfaces the per-track discard reasons when the conversion is invalid', async () => {
    conversionResult = invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'max_track_count_of_type_reached' }])

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/video track #1 \(codec: hevc\): max_track_count_of_type_reached/)
    expect(executeMock).not.toHaveBeenCalled()
  })

  it('blames missing WebCodecs support when the API itself is unavailable', async () => {
    // The software decoder can't stand in here: it replaces only the decode
    // step, and encoding still needs WebCodecs.
    vi.stubGlobal('VideoDecoder', undefined)
    vi.stubGlobal('AudioDecoder', undefined)
    conversionResult = invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' }])

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/WebCodecs APIs this app needs/)
  })

  it('names the insecure connection when that is why WebCodecs is missing', async () => {
    vi.stubGlobal('VideoDecoder', undefined)
    vi.stubGlobal('AudioDecoder', undefined)
    vi.stubGlobal('self', { isSecureContext: false })
    conversionResult = invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' }])

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/insecure connection/)
  })
})

describe('renderVideo frame compositing', () => {
  /** Pulls the `process` callback out of the Conversion.init options and runs
   * it against a stand-in sample, which is the only way to observe what the
   * HUD is actually asked to draw per frame. */
  async function runProcess(options: { trimStart: number; trimEnd: number }, sampleTimestamp: number) {
    const drawn: unknown[] = []
    await renderVideo({} as File, { ...baseOptions, ...options, annotatedRows: [{ ...row, time: options.trimStart }] })
    const initOptions = initMock.mock.calls[0][0] as { video: { process: (s: unknown) => unknown } }
    const sample = {
      timestamp: sampleTimestamp,
      draw: (...args: unknown[]) => drawn.push(args),
    }
    const result = initOptions.video.process(sample)
    return { drawn, result }
  }

  it('draws the video frame before the HUD, into the same canvas', async () => {
    const { drawn, result } = await runProcess({ trimStart: 0, trimEnd: 10 }, 1)

    expect(drawn).toHaveLength(1)
    expect(hudDraws).toHaveLength(1)
    expect(result).toBeInstanceOf(FakeOffscreenCanvas)
  })

  it('tells the HUD not to clear the canvas', async () => {
    // HudRenderer wipes the canvas by default, which erased the video frame
    // that had just been drawn -- the render came out as a black picture with
    // only the HUD on it.
    await runProcess({ trimStart: 0, trimEnd: 10 }, 1)

    expect(hudDraws[0].clear).toBe(false)
  })

  it('asks the HUD for absolute time, not the trim-rebased timestamp', async () => {
    // Conversion rebases sample timestamps to the trim window before calling
    // `process`, but the telemetry rows keep their original absolute times.
    // Passing the rebased value straight through asked for telemetry from
    // before the window began, and the nearest-row search clamps -- so the
    // HUD froze on row 0 for the entire video.
    await runProcess({ trimStart: 120, trimEnd: 180 }, 5)

    expect(hudDraws[0].time).toBe(125)
  })

  it('keeps HUD and footage aligned when the trim starts at zero', async () => {
    await runProcess({ trimStart: 0, trimEnd: 60 }, 12.5)

    expect(hudDraws[0].time).toBe(12.5)
  })
})
