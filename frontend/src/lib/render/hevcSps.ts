/**
 * Just enough HEVC sequence-parameter-set parsing to learn how far the stream
 * reorders frames.
 *
 * The software decoder (hevcDecoder.ts) has to hand mediabunny frames in
 * *presentation* order, the same contract WebCodecs' `VideoDecoder` follows.
 * libde265 gives them back in *decode* order, which for any stream with
 * B-frames is a different order entirely -- so frames have to be buffered and
 * re-sorted before they go out.
 *
 * `sps_max_num_reorder_pics` is exactly the number that says how deep that
 * buffer must be: the standard guarantees no more than that many pictures
 * precede a given picture in decode order while following it in output order.
 * Buffering that many and always emitting the lowest timestamp is therefore
 * correct, not a heuristic -- and it's much cheaper than the alternative of
 * holding a whole GOP, which at 4K would be hundreds of megabytes.
 */

/** Strips emulation-prevention bytes: inside a NAL, the encoder inserts a 0x03
 * after any 0x0000 that would otherwise look like a start code. Reading the
 * bitstream without removing them again misaligns every field after the first
 * occurrence. */
function unescapeRbsp(nal: Uint8Array): Uint8Array {
  const out = new Uint8Array(nal.length)
  let written = 0
  for (let i = 0; i < nal.length; i++) {
    if (i >= 2 && nal[i] === 3 && nal[i - 1] === 0 && nal[i - 2] === 0) continue
    out[written++] = nal[i]
  }
  return out.subarray(0, written)
}

/** Big-endian bit reader with the unsigned Exp-Golomb decoding H.265 uses for
 * most of its variable-length fields. */
class BitReader {
  private pos = 0
  private readonly bytes: Uint8Array

  constructor(bytes: Uint8Array) {
    this.bytes = bytes
  }

  bit(): number {
    if (this.pos >> 3 >= this.bytes.length) throw new Error('ran past the end of the bitstream')
    const bit = (this.bytes[this.pos >> 3] >> (7 - (this.pos & 7))) & 1
    this.pos++
    return bit
  }

  u(count: number): number {
    let value = 0
    for (let i = 0; i < count; i++) value = value * 2 + this.bit()
    return value
  }

  /** ue(v) -- unsigned Exp-Golomb. */
  ue(): number {
    let leadingZeros = 0
    while (this.bit() === 0) {
      leadingZeros++
      if (leadingZeros > 31) throw new Error('malformed Exp-Golomb code')
    }
    if (leadingZeros === 0) return 0
    return 2 ** leadingZeros - 1 + this.u(leadingZeros)
  }
}

/** `profile_tier_level` is a fixed-size block we only need to step over to
 * reach the fields after it. With a profile present that's 8 bits of
 * space/tier/idc, 32 compatibility flags, 48 bits of constraint and reserved
 * flags, then the level -- followed by optional per-sub-layer repeats. */
function skipProfileTierLevel(reader: BitReader, maxSubLayersMinus1: number): void {
  reader.u(8)
  for (let i = 0; i < 32; i++) reader.bit()
  for (let i = 0; i < 48; i++) reader.bit()
  reader.u(8)

  const profilePresent: number[] = []
  const levelPresent: number[] = []
  for (let i = 0; i < maxSubLayersMinus1; i++) {
    profilePresent.push(reader.bit())
    levelPresent.push(reader.bit())
  }
  // The sub-layer flags are padded out to eight entries whenever there is more
  // than one sub-layer.
  if (maxSubLayersMinus1 > 0) for (let i = maxSubLayersMinus1; i < 8; i++) reader.u(2)
  for (let i = 0; i < maxSubLayersMinus1; i++) {
    if (profilePresent[i]) {
      reader.u(8)
      for (let j = 0; j < 32; j++) reader.bit()
      for (let j = 0; j < 48; j++) reader.bit()
    }
    if (levelPresent[i]) reader.u(8)
  }
}

export interface HevcSpsInfo {
  /** How many frames must be buffered to restore presentation order. */
  maxNumReorderPics: number
  /** `chroma_format_idc`: 1 is 4:2:0, which is what action cams record. */
  chromaFormatIdc: number
  bitDepthLuma: number
}

/**
 * Reads the fields we need out of one SPS NAL unit (including its two-byte NAL
 * header). Throws if the bitstream doesn't parse.
 */
export function parseHevcSps(spsNal: Uint8Array): HevcSpsInfo {
  const reader = new BitReader(unescapeRbsp(spsNal.subarray(2)))

  reader.u(4) // sps_video_parameter_set_id
  const maxSubLayersMinus1 = reader.u(3)
  reader.bit() // sps_temporal_id_nesting_flag
  skipProfileTierLevel(reader, maxSubLayersMinus1)

  reader.ue() // sps_seq_parameter_set_id
  const chromaFormatIdc = reader.ue()
  if (chromaFormatIdc === 3) reader.bit() // separate_colour_plane_flag
  reader.ue() // pic_width_in_luma_samples
  reader.ue() // pic_height_in_luma_samples
  if (reader.bit()) {
    // conformance_window_flag: four offsets we don't need, since libde265
    // applies the crop itself before handing frames back.
    reader.ue()
    reader.ue()
    reader.ue()
    reader.ue()
  }
  const bitDepthLuma = reader.ue() + 8
  reader.ue() // bit_depth_chroma_minus8
  reader.ue() // log2_max_pic_order_cnt_lsb_minus4

  const subLayerOrderingInfoPresent = reader.bit()
  let maxNumReorderPics = 0
  // When the per-sub-layer flag is clear, only the highest sub-layer's values
  // are coded and they apply throughout.
  for (let i = subLayerOrderingInfoPresent ? 0 : maxSubLayersMinus1; i <= maxSubLayersMinus1; i++) {
    reader.ue() // sps_max_dec_pic_buffering_minus1
    maxNumReorderPics = Math.max(maxNumReorderPics, reader.ue())
    reader.ue() // sps_max_latency_increase_plus1
  }

  return { maxNumReorderPics, chromaFormatIdc, bitDepthLuma }
}

/** `nal_unit_type` 33 is SPS_NUT. */
export const NAL_UNIT_TYPE_SPS = 33
