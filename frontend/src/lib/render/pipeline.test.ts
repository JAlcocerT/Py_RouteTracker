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

beforeEach(() => {
  executeMock.mockClear()
})

describe('renderVideo', () => {
  it('throws a message naming the discarded track and reason instead of calling execute', async () => {
    // A mixed-reason discard, not all 'undecodable_source_codec' -- this
    // should hit the general per-track message, not the WebCodecs-outage
    // special case (covered separately below).
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

  it('blames the real cause -- no WebCodecs support -- when every track is discarded as undecodable', async () => {
    // Reproduces a real report: every track (video AND an ordinarily
    // universally-decodable audio codec like aac) coming back
    // 'undecodable_source_codec' isn't really a per-codec gap -- it's
    // VideoDecoder/AudioDecoder being entirely absent (e.g. an insecure
    // context, like this app's own documented plain-http Tailscale URL).
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
    ).rejects.toThrow(/WebCodecs/)
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
