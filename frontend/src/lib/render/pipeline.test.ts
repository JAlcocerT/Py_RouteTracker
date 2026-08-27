import { beforeEach, describe, expect, it, vi } from 'vitest'
import { renderVideo } from './pipeline'
import { DEFAULT_RENDER_CONFIG } from './renderConfig'

// Mediabunny's Conversion talks to real codecs/WebCodecs, which don't exist
// in this test environment -- mocked at the module boundary so this only
// exercises renderVideo's own isValid/discardedTracks handling (the bug this
// test guards: a video whose track(s) got discarded used to fail deep inside
// conversion.execute() with mediabunny's own generic message instead of
// renderVideo surfacing which track and why beforehand).
interface FakeConversion {
  isValid: boolean
  discardedTracks: { track: { type: string; number: number; codec: string }; reason: string }[]
  execute: () => Promise<void>
}

const primaryVideoTrack = { getDisplayWidth: vi.fn(async () => 640), getDisplayHeight: vi.fn(async () => 360) }
const executeMock = vi.fn(async () => undefined)
let conversionInit: () => Promise<FakeConversion>

vi.mock('mediabunny', () => ({
  ALL_FORMATS: [],
  BlobSource: vi.fn(),
  BufferTarget: vi.fn(function (this: { buffer: ArrayBuffer }) {
    this.buffer = new ArrayBuffer(0)
  }),
  StreamTarget: vi.fn(),
  Mp4OutputFormat: vi.fn(),
  Output: vi.fn(),
  Input: vi.fn(function (this: { getPrimaryVideoTrack: () => Promise<typeof primaryVideoTrack> }) {
    this.getPrimaryVideoTrack = async () => primaryVideoTrack
  }),
  Conversion: { init: () => conversionInit() },
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

const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

beforeEach(() => {
  executeMock.mockClear()
  consoleErrorSpy.mockClear()
  // Present by default (WebCodecs itself works) -- the one test that needs
  // the API genuinely missing overrides this for itself.
  vi.stubGlobal('VideoDecoder', class {})
  vi.stubGlobal('AudioDecoder', class {})
})

describe('renderVideo', () => {
  it('throws a message naming the discarded track and reason for a non-codec discard', async () => {
    conversionInit = vi.fn(async () => ({
      isValid: false,
      discardedTracks: [{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'max_track_count_of_type_reached' }],
      execute: executeMock,
    }))

    await expect(
      renderVideo({} as File, { trimStart: 0, trimEnd: 1, config: DEFAULT_RENDER_CONFIG, annotatedRows: [row] }),
    ).rejects.toThrow(/video track #1 \(codec: hevc\): max_track_count_of_type_reached/)
    expect(executeMock).not.toHaveBeenCalled()
  })

  it('logs per-track diagnostics and points at the console when tracks are undecodable but WebCodecs itself works', async () => {
    // Reproduces a real report: video (hevc) AND audio (an ordinarily
    // near-universally-decodable codec, aac) both coming back
    // 'undecodable_source_codec' together, even though VideoDecoder/
    // AudioDecoder are present and functional -- an earlier version of this
    // message guessed "licensed codec support gap", which didn't hold up
    // against this exact report on an official Chrome install. Rather than
    // guess a specific cause again, this should surface the real decoder
    // config for diagnosis instead.
    conversionInit = vi.fn(async () => ({
      isValid: false,
      discardedTracks: [
        { track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' },
        { track: { type: 'audio', number: 1, codec: 'aac' }, reason: 'undecodable_source_codec' },
      ],
      execute: executeMock,
    }))

    await expect(
      renderVideo({} as File, { trimStart: 0, trimEnd: 1, config: DEFAULT_RENDER_CONFIG, annotatedRows: [row] }),
    ).rejects.toThrow(
      /video track #1 \(codec: hevc\): undecodable_source_codec.*audio track #1 \(codec: aac\): undecodable_source_codec.*open the browser console/,
    )
    expect(executeMock).not.toHaveBeenCalled()
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining('discarded tracks'),
      expect.arrayContaining([expect.objectContaining({ type: 'video', codec: 'hevc' }), expect.objectContaining({ type: 'audio', codec: 'aac' })]),
      expect.anything(),
      expect.anything(),
    )
  })

  it('blames missing WebCodecs support, not codec licensing, when the API itself is unavailable', async () => {
    vi.stubGlobal('VideoDecoder', undefined)
    vi.stubGlobal('AudioDecoder', undefined)
    conversionInit = vi.fn(async () => ({
      isValid: false,
      discardedTracks: [{ track: { type: 'video', number: 1, codec: 'hevc' }, reason: 'undecodable_source_codec' }],
      execute: executeMock,
    }))

    await expect(
      renderVideo({} as File, { trimStart: 0, trimEnd: 1, config: DEFAULT_RENDER_CONFIG, annotatedRows: [row] }),
    ).rejects.toThrow(/WebCodecs APIs this app needs/)
    expect(executeMock).not.toHaveBeenCalled()
  })

  it('proceeds to execute when the conversion is valid', async () => {
    conversionInit = vi.fn(async () => ({
      isValid: true,
      discardedTracks: [],
      execute: executeMock,
    }))

    const result = await renderVideo({} as File, { trimStart: 0, trimEnd: 1, config: DEFAULT_RENDER_CONFIG, annotatedRows: [row] })

    expect(executeMock).toHaveBeenCalledTimes(1)
    expect(result).toBeInstanceOf(Blob)
  })
})
