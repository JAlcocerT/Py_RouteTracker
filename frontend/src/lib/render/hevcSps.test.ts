import { describe, expect, it } from 'vitest'
import { parseHevcSps } from './hevcSps'

/**
 * Real SPS NAL units, lifted from the hvcC box of files produced by x265.
 * Synthetic bitstreams would only test the parser against my own understanding
 * of the syntax; these test it against what an encoder actually emits.
 *
 * Each was cross-checked against an independent decode of the same file --
 * libde265 reports the same chroma format and bit depth these parse out.
 */
const NO_B_FRAMES = Uint8Array.from([
  66, 1, 1, 1, 96, 0, 0, 3, 0, 144, 0, 0, 3, 0, 0, 3, 0, 60, 160, 10, 8, 15, 22, 89, 42, 73, 50, 188, 5, 160, 32, 0, 0, 3, 0, 32, 0, 0, 3, 3, 193,
])
const WITH_B_FRAMES = Uint8Array.from([
  66, 1, 1, 1, 96, 0, 0, 3, 0, 144, 0, 0, 3, 0, 0, 3, 0, 60, 160, 10, 8, 15, 22, 89, 88, 169, 36, 202, 240, 22, 128, 128, 0, 0, 3, 0, 128, 0, 0, 15,
  4,
])
const MAIN_10 = Uint8Array.from([
  66, 1, 1, 2, 32, 0, 0, 3, 0, 144, 0, 0, 3, 0, 0, 3, 0, 120, 160, 3, 192, 128, 16, 228, 217, 101, 102, 146, 76, 175, 1, 104, 8, 0, 0, 3, 0, 8, 0, 0,
  3, 0, 240, 64,
])

describe('parseHevcSps', () => {
  it('reports no reordering for a stream encoded without B-frames', () => {
    // The depth that matters most to get right: buffering frames that never
    // need reordering would delay every frame of the render for nothing.
    expect(parseHevcSps(NO_B_FRAMES).maxNumReorderPics).toBe(0)
  })

  it('reports the real reorder depth for a stream with B-frames', () => {
    // Decode order and presentation order genuinely differ here, so this is
    // the number that keeps the output file's frames in the right sequence.
    expect(parseHevcSps(WITH_B_FRAMES).maxNumReorderPics).toBe(2)
  })

  it('reads chroma format and bit depth for 8-bit Main', () => {
    expect(parseHevcSps(WITH_B_FRAMES)).toMatchObject({ chromaFormatIdc: 1, bitDepthLuma: 8 })
  })

  it('reads the bit depth of a Main 10 stream', () => {
    // Newer GoPros record 10-bit at higher bitrates, and it decodes to a
    // different pixel format (I420P10), so this must not be mistaken for 8-bit.
    expect(parseHevcSps(MAIN_10)).toMatchObject({ chromaFormatIdc: 1, bitDepthLuma: 10 })
  })

  it('throws rather than returning nonsense when the NAL is truncated', () => {
    // hevcDecoder.ts treats a parse failure as "assume no reordering", which
    // is only safe if failure is loud rather than a silently wrong number.
    expect(() => parseHevcSps(WITH_B_FRAMES.subarray(0, 6))).toThrow()
  })

  it('undoes emulation-prevention bytes', () => {
    // These fixtures all contain 0x000003 sequences (visible as `0, 0, 3` in
    // the arrays above). Reading straight past them would misalign every field
    // that follows, so correct parses here are what proves the unescaping.
    expect(NO_B_FRAMES).toContain(3)
    expect(parseHevcSps(NO_B_FRAMES).chromaFormatIdc).toBe(1)
  })
})
