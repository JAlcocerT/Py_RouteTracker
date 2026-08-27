import { afterEach, describe, expect, it, vi } from 'vitest'
import { describeCodecCompatIssue, hasWebCodecsSupport, isInsecureContext } from './env'

const canDecodeVideo = vi.fn()
const canDecodeAudio = vi.fn()
vi.mock('mediabunny', () => ({ canDecodeVideo: (...args: unknown[]) => canDecodeVideo(...args), canDecodeAudio: (...args: unknown[]) => canDecodeAudio(...args) }))

describe('hasWebCodecsSupport', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('is false when VideoDecoder/AudioDecoder are unavailable (jsdom, or an insecure-context browser)', () => {
    expect(hasWebCodecsSupport()).toBe(false)
  })

  it('is true once both decoders are present', () => {
    vi.stubGlobal('VideoDecoder', class {})
    vi.stubGlobal('AudioDecoder', class {})
    expect(hasWebCodecsSupport()).toBe(true)
  })
})

describe('isInsecureContext', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('is true when self.isSecureContext is explicitly false', () => {
    vi.stubGlobal('self', { isSecureContext: false })
    expect(isInsecureContext()).toBe(true)
  })

  it('is false when self.isSecureContext is true', () => {
    vi.stubGlobal('self', { isSecureContext: true })
    expect(isInsecureContext()).toBe(false)
  })
})

describe('describeCodecCompatIssue', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('blames missing WebCodecs support when the decoder classes are absent', async () => {
    vi.stubGlobal('self', { isSecureContext: true })
    await expect(describeCodecCompatIssue()).resolves.toMatch(/doesn't support the video decoding\/encoding APIs/)
  })

  it('blames the insecure connection specifically when that is the cause', async () => {
    vi.stubGlobal('self', { isSecureContext: false })
    await expect(describeCodecCompatIssue()).resolves.toMatch(/insecure connection/)
  })

  it('flags a real GoPro-style codec gap (hevc/aac undecodable) even though WebCodecs itself works', async () => {
    vi.stubGlobal('self', { isSecureContext: true })
    vi.stubGlobal('VideoDecoder', class {})
    vi.stubGlobal('AudioDecoder', class {})
    canDecodeVideo.mockImplementation(async (codec: string) => codec === 'avc')
    canDecodeAudio.mockImplementation(async () => false)

    await expect(describeCodecCompatIssue()).resolves.toMatch(/HEVC \(H\.265\)/)
  })

  it('is null when h264/hevc/aac all decode fine', async () => {
    vi.stubGlobal('self', { isSecureContext: true })
    vi.stubGlobal('VideoDecoder', class {})
    vi.stubGlobal('AudioDecoder', class {})
    canDecodeVideo.mockResolvedValue(true)
    canDecodeAudio.mockResolvedValue(true)

    await expect(describeCodecCompatIssue()).resolves.toBeNull()
  })
})
