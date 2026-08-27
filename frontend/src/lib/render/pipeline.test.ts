import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderVideo } from './pipeline'
import { DEFAULT_RENDER_CONFIG } from './renderConfig'

// Mediabunny's Conversion talks to real codecs/WebCodecs, which don't exist
// in this test environment -- mocked at the module boundary so this only
// exercises renderVideo's own isValid/discardedTracks handling and its
// software-decode fallback, not the codecs themselves.
interface FakeConversion {
  isValid: boolean
  discardedTracks: { track: { type: string; number: number; codec: string }; reason: string }[]
  execute: () => Promise<void>
}

const primaryVideoTrack = { getDisplayWidth: vi.fn(async () => 640), getDisplayHeight: vi.fn(async () => 360) }
const executeMock = vi.fn(async () => undefined)
/** One entry per expected Conversion.init call, so the fallback path can be
 * given a failing first pass and a succeeding second one. */
let conversionQueue: FakeConversion[] = []

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
    init: () => {
      const next = conversionQueue.shift()
      if (!next) throw new Error('Conversion.init called more times than the test queued results for')
      return Promise.resolve(next)
    },
  },
}))

// ffmpeg.wasm can't run under vitest ("does not support nodejs"), and this
// suite is about *when* the fallback fires, not the transcode itself.
const transcodeMock = vi.fn(async () => new File([], 'compat.webm', { type: 'video/webm' }))
vi.mock('./wasmTranscode', () => ({ transcodeForCompatibility: (...args: unknown[]) => transcodeMock(...(args as [])) }))

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
  executeMock.mockClear()
  transcodeMock.mockClear()
  conversionQueue = []
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  // Present by default (WebCodecs itself works) -- the test that needs the
  // API genuinely missing overrides this for itself.
  vi.stubGlobal('VideoDecoder', class {})
  vi.stubGlobal('AudioDecoder', class {})
})

describe('renderVideo', () => {
  it('reports a non-codec discard without attempting a software transcode', async () => {
    // Re-encoding the input can't create room in the output container, so
    // spending minutes of CPU to reach the same failure would be worse than
    // failing now.
    conversionQueue = [invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'max_track_count_of_type_reached' }])]

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/video track #1 \(codec: hevc\): max_track_count_of_type_reached/)
    expect(transcodeMock).not.toHaveBeenCalled()
    expect(executeMock).not.toHaveBeenCalled()
  })

  it('falls back to a software transcode when the browser cannot decode the source codecs', async () => {
    // The real report this fallback exists for: hevc video and aac audio
    // both discarded as undecodable, on a browser whose WebCodecs API is
    // present and working -- so the codecs, not the API, are what's missing.
    conversionQueue = [
      invalidWith([
        { track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' },
        { track: { type: 'audio', number: 1, codec: 'aac' }, reason: 'undecodable_source_codec' },
      ]),
      valid(),
    ]

    const result = await renderVideo({} as File, { ...baseOptions, trimStart: 5, trimEnd: 9, annotatedRows: [{ ...row, time: 6 }] })

    expect(transcodeMock).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ trimStart: 5, trimEnd: 9 }))
    expect(executeMock).toHaveBeenCalledTimes(1)
    expect(result).toBeInstanceOf(Blob)
  })

  it('surfaces the original error rather than looping when the transcoded file is still undecodable', async () => {
    conversionQueue = [
      invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' }]),
      invalidWith([{ track: { type: 'video', number: 1, codec: 'vp8' }, reason: 'undecodable_source_codec' }]),
    ]

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/codec: vp8/)
    // Exactly one attempt -- the retry must not itself retry.
    expect(transcodeMock).toHaveBeenCalledTimes(1)
  })

  it('blames missing WebCodecs support instead of transcoding, when the API itself is unavailable', async () => {
    // Transcoding produces VP8, which pass two would still decode through
    // WebCodecs -- with no API at all there is nothing to fall back to.
    vi.stubGlobal('VideoDecoder', undefined)
    vi.stubGlobal('AudioDecoder', undefined)
    conversionQueue = [invalidWith([{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' }])]

    await expect(renderVideo({} as File, baseOptions)).rejects.toThrow(/WebCodecs APIs this app needs/)
    expect(transcodeMock).not.toHaveBeenCalled()
  })

  it('proceeds to execute when the conversion is valid', async () => {
    conversionQueue = [valid()]

    const result = await renderVideo({} as File, baseOptions)

    expect(executeMock).toHaveBeenCalledTimes(1)
    expect(transcodeMock).not.toHaveBeenCalled()
    expect(result).toBeInstanceOf(Blob)
  })
})
