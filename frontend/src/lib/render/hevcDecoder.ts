/**
 * Software HEVC decoding, for the (very common) case where the browser's own
 * decoders can't do it.
 *
 * Chrome and Edge expose HEVC through WebCodecs *only* when the platform hands
 * them a hardware decoder -- VideoToolbox on macOS, Media Foundation (plus the
 * "HEVC Video Extensions") on Windows, VAAPI on Linux. Chromium ships no
 * software HEVC decoder at all, so on a Linux box with an NVIDIA/AMD GPU, in a
 * VM, or on a Windows install without the HEVC extension, `VideoDecoder`
 * rejects the config outright and no amount of configuration changes that.
 * Newer GoPros record HEVC by default at higher resolutions/frame rates, so
 * this is the normal case for exactly the footage this app exists to process.
 *
 * The fix is to bring our own decoder: libde265, compiled to WebAssembly. It
 * doesn't care what the browser shipped.
 *
 * What makes this cheap rather than a second pipeline: mediabunny lets a
 * decoder be *registered* (`registerDecoder`), after which its normal
 * `Conversion` path -- trim, the HUD `process` callback, audio passthrough,
 * MP4 muxing -- runs completely unchanged, and the *encode* still goes through
 * the browser's own (hardware) H.264 encoder. So this replaces one step, not
 * the pipeline: one software decode, streamed frame by frame with bounded
 * memory, instead of the whole-file transcode-to-an-intermediate this replaced
 * (see git history for `wasmTranscode.ts`, which had to work that way because
 * ffmpeg.wasm's `exec(argv)` API cannot stream packets).
 *
 * Measured throughput of this build, decoding x265-encoded test footage:
 * ~45fps at 1080p 8-bit, ~38fps at 1080p 10-bit, ~14fps at 4K 8-bit. Output
 * was verified bit-exact against ffmpeg's own decoder. libde265's WASM build
 * is single-threaded, so those numbers are per core and don't improve with
 * cross-origin isolation.
 */
import { CustomVideoDecoder, VideoSample, registerDecoder } from 'mediabunny'
import type { EncodedPacket, VideoSamplePixelFormat } from 'mediabunny'
import initLibde265 from '@yume-chan/libde265'
import type { Decoder, MainModule } from '@yume-chan/libde265'
import { NAL_UNIT_TYPE_SPS, parseHevcSps } from './hevcSps'
// Vite emits the wasm as a build asset and hands back its final hashed URL,
// which is then given to emscripten's `locateFile`. Without this the module
// would try to resolve the file relative to its own bundled location and 404.
import libde265WasmUrl from '@yume-chan/libde265/libde265.wasm?url'

/** libde265's `de265_chroma` for 4:2:0. The only chroma format action-cam
 * footage uses in practice -- 4:2:2/4:4:4 HEVC exists but belongs to cinema
 * cameras and screen capture. */
const CHROMA_420 = 1

/**
 * libde265 reports colour signalling as raw H.273 code points, while
 * `VideoSample` wants WebCodecs' string enums. Only the values WebCodecs
 * actually defines are mapped; anything else yields `undefined`, which leaves
 * the sample unannotated rather than mislabelled -- guessing here would shift
 * the colours of the rendered footage.
 *
 * Action-cam footage is overwhelmingly BT.709; the BT.2020/PQ/HLG entries are
 * what HDR GoPro modes produce.
 *
 * Typed as plain strings because TypeScript's bundled `lib.dom.d.ts` still
 * describes an older WebCodecs revision whose colour enums stop at
 * smpte170m -- browsers accept the HDR values these maps emit, the type
 * definitions just haven't caught up. Hence the single assertion where the
 * colour space is assembled, rather than dropping HDR annotation entirely.
 */
const PRIMARIES: Record<number, string | undefined> = {
  1: 'bt709',
  5: 'bt470bg',
  6: 'smpte170m',
  9: 'bt2020',
  12: 'smpte432',
}
const TRANSFER: Record<number, string | undefined> = {
  1: 'bt709',
  6: 'smpte170m',
  8: 'linear',
  13: 'iec61966-2-1',
  16: 'pq',
  18: 'hlg',
}
const MATRIX: Record<number, string | undefined> = {
  0: 'rgb',
  1: 'bt709',
  5: 'bt470bg',
  6: 'smpte170m',
  9: 'bt2020-ncl',
}

/** Frame timestamps are passed through libde265 as microseconds: it doesn't
 * interpret them at all, it just hands them back on the decoded image, so this
 * only has to be fine enough that no real frame rate rounds onto its neighbour
 * and integral so nothing drifts over a long render. */
const MICROSECONDS_PER_SECOND = 1_000_000

/**
 * Whether *this* build of the browser can decode a given HEVC config natively.
 *
 * Deliberately probed with the real track config rather than a representative
 * one: HEVC support is frequently partial (a decoder that takes 8-bit Main at
 * 1080p may still refuse Main10, or 4K, or a level it has no hardware for), so
 * a generic probe would answer a different question than the one being asked.
 */
async function canDecodeNatively(config: VideoDecoderConfig): Promise<boolean> {
  if (typeof VideoDecoder === 'undefined') return false
  try {
    const support = await VideoDecoder.isConfigSupported(config)
    return support.supported === true
  } catch {
    // A config the browser can't even parse is, for our purposes, one it
    // can't decode.
    return false
  }
}

function isHevc(codec: string): boolean {
  // 'hvc1' and 'hev1' differ only in whether parameter sets live in the sample
  // entry or in-band; both are HEVC and both turn up in action-cam MP4s.
  const lower = codec.toLowerCase()
  return lower.startsWith('hvc1') || lower.startsWith('hev1') || lower === 'hevc'
}

/** Registration is global and permanent, so it must happen at most once --
 * and `supports()` below is synchronous while the native probe is not, hence
 * resolving the question here and having `supports()` read the answer. */
let registered = false

/**
 * Registers the software HEVC decoder with mediabunny, but only if this
 * browser can't decode `config` natively.
 *
 * The gate matters in both directions. Registering when hardware decode is
 * available would force every machine onto a software path that is an order of
 * magnitude slower, for no reason. Not registering when it isn't available
 * fails the render outright. Returns whether the software path is now in play,
 * so callers can tell the user why this render is slow.
 */
export async function ensureSoftwareHevcDecoder(config: VideoDecoderConfig | null): Promise<boolean> {
  if (!config || !isHevc(config.codec)) return false
  if (await canDecodeNatively(config)) return false
  if (!registered) {
    registerDecoder(WasmHevcDecoder)
    registered = true
  }
  return true
}

/** Shared across renders in this worker: fetching and compiling the wasm
 * shouldn't be paid for twice in one session. Each decoder still gets its own
 * `Decoder` instance from it. */
let modulePromise: Promise<MainModule> | null = null

function loadLibde265(): Promise<MainModule> {
  if (!modulePromise) {
    modulePromise = initLibde265({ locateFile: () => libde265WasmUrl })
    // A failed load must not stay cached, or every later attempt in this
    // session resolves to the same rejection with no way to retry.
    modulePromise.catch(() => {
      modulePromise = null
    })
  }
  return modulePromise
}

/**
 * Pulls the parameter-set NAL units (VPS/SPS/PPS) and the NAL length prefix
 * size out of an hvcC box -- the `description` mediabunny reads from the MP4
 * sample entry.
 *
 * libde265 wants NAL units, whereas MP4 stores them length-prefixed with the
 * parameter sets hoisted out into this box. So the parameter sets are replayed
 * into the decoder up front, and `nalLengthSize` is what lets `decode()` split
 * each sample back into the NALs it's made of.
 */
function parseHvcC(description: AllowSharedBufferSource): { parameterSets: Uint8Array[]; spsNals: Uint8Array[]; nalLengthSize: number } {
  const d = ArrayBuffer.isView(description)
    ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
    : new Uint8Array(description)

  // Fixed-layout header: 21 bytes of profile/level/format fields, then a byte
  // whose low 2 bits are lengthSizeMinusOne, then the array count.
  if (d.byteLength < 23) throw new Error('HEVC configuration box is too short to be valid')
  const nalLengthSize = (d[21] & 0b11) + 1
  const numArrays = d[22]

  const parameterSets: Uint8Array[] = []
  const spsNals: Uint8Array[] = []
  let offset = 23
  for (let i = 0; i < numArrays; i++) {
    // Each array: 1 byte of flags+NAL type, then a uint16 count, then that
    // many uint16-length-prefixed NAL units.
    if (offset + 3 > d.byteLength) break
    const nalUnitType = d[offset] & 0b111111
    const numNalus = (d[offset + 1] << 8) | d[offset + 2]
    offset += 3
    for (let j = 0; j < numNalus; j++) {
      if (offset + 2 > d.byteLength) break
      const length = (d[offset] << 8) | d[offset + 1]
      const nal = d.subarray(offset + 2, offset + 2 + length)
      parameterSets.push(nal)
      if (nalUnitType === NAL_UNIT_TYPE_SPS) spsNals.push(nal)
      offset += 2 + length
    }
  }
  return { parameterSets, spsNals, nalLengthSize }
}

export class WasmHevcDecoder extends CustomVideoDecoder {
  /** Unconditionally true for HEVC: `ensureSoftwareHevcDecoder` above has
   * already established that the native decoder can't handle this file, and
   * registration only happens in that case. mediabunny consults registered
   * decoders *before* `VideoDecoder` (see its input-track.js `canDecode`), so
   * a broader answer here would shadow working hardware decoders. */
  static supports(codec: string): boolean {
    return codec === 'hevc'
  }

  private module: MainModule | null = null
  private decoder: Decoder | null = null
  private parameterSets: Uint8Array[] = []
  private nalLengthSize = 4
  /** How deep this stream reorders; see hevcSps.ts and `reorder` below. */
  private maxNumReorderPics = 0
  /** Decoded frames held back to restore presentation order, kept sorted by
   * timestamp so the one to emit next is always at the front. */
  private pending: VideoSample[] = []

  async init(): Promise<void> {
    if (!this.config.description) {
      // Without the hvcC box there are no parameter sets to prime the decoder
      // with. In-band parameter sets ('hev1') would work, but every MP4 this
      // app sees carries them out-of-band, so failing clearly beats emitting
      // green frames.
      throw new Error('This HEVC file has no codec configuration (hvcC) box, which the software decoder needs to start')
    }
    const { parameterSets, spsNals, nalLengthSize } = parseHvcC(this.config.description)
    this.parameterSets = parameterSets
    this.nalLengthSize = nalLengthSize

    // A stream whose SPS won't parse still decodes fine; it just doesn't tell
    // us how far it reorders. Falling back to zero keeps such a file playable
    // (correct for the many action-cam streams recorded without B-frames)
    // rather than failing the render outright over a diagnostic detail.
    for (const sps of spsNals) {
      try {
        this.maxNumReorderPics = Math.max(this.maxNumReorderPics, parseHevcSps(sps).maxNumReorderPics)
      } catch {
        // best-effort, see above
      }
    }

    this.module = await loadLibde265()
    this.decoder = new this.module.Decoder()
    this.primeParameterSets()
  }

  private primeParameterSets(): void {
    for (const nal of this.parameterSets) this.decoder!.pushNal(nal, 0n)
  }

  async decode(packet: EncodedPacket): Promise<void> {
    const decoder = this.decoder
    if (!decoder) throw new Error('HEVC decoder used before init()')

    const pts = BigInt(Math.round(packet.timestamp * MICROSECONDS_PER_SECOND))
    const data = packet.data
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)

    // One MP4 sample is a run of length-prefixed NAL units, not a single one.
    let offset = 0
    while (offset + this.nalLengthSize <= data.byteLength) {
      let length = 0
      for (let i = 0; i < this.nalLengthSize; i++) length = length * 256 + view.getUint8(offset + i)
      offset += this.nalLengthSize
      if (length === 0 || offset + length > data.byteLength) break
      decoder.pushNal(data.subarray(offset, offset + length), pts)
      offset += length
    }
    // Marks the access unit complete. Without it libde265 waits to see where
    // the next one starts, and the final frame of the stream never comes out
    // at all -- but it also makes libde265 emit pictures as soon as they are
    // decoded, in decode order, which is why `reorder` below exists.
    decoder.pushEndOfFrame()
    this.drain()
    this.emitReordered(this.maxNumReorderPics)
  }

  async flush(): Promise<void> {
    const decoder = this.decoder
    if (!decoder) return
    decoder.flushData()
    this.drain()
    // Nothing more is coming, so the reorder buffer can be emptied completely.
    this.emitReordered(0)
    // Leave the decoder usable: mediabunny flushes at seek boundaries as well
    // as at end of stream, and a reset one needs its parameter sets again
    // before it can decode anything further.
    decoder.reset()
    this.primeParameterSets()
  }

  /**
   * Emits buffered frames, oldest first, until at most `keep` remain.
   *
   * Holding back `sps_max_num_reorder_pics` frames is what converts libde265's
   * decode-order output into the presentation order mediabunny expects, and
   * the standard guarantees that depth is sufficient -- so this restores the
   * correct order rather than approximating it.
   */
  private emitReordered(keep: number): void {
    while (this.pending.length > keep) {
      this.onSample(this.pending.shift()!)
    }
  }

  /** Inserts into the pending buffer keeping it sorted by timestamp. The
   * buffer is only a handful of frames deep, so a linear insert beats the
   * bookkeeping a heap would need. */
  private buffer(sample: VideoSample): void {
    let i = this.pending.length
    while (i > 0 && this.pending[i - 1].timestamp > sample.timestamp) i--
    this.pending.splice(i, 0, sample)
  }

  /** Pulls out every frame the decoder can currently produce, into the
   * reorder buffer. */
  private drain(): void {
    const module = this.module!
    const decoder = this.decoder!

    let more = true
    while (more) {
      const result = decoder.decode()
      more = result.more

      // Collect before reacting to the status: at end of stream the last
      // picture becomes available on the very call that reports
      // WAITING_FOR_INPUT_DATA, so checking the status first would drop it.
      for (;;) {
        const image = decoder.getNextPicture()
        if (!image) break
        try {
          this.buffer(this.toVideoSample(image))
        } finally {
          // Mandatory: libde265 hands out a fixed pool of image buffers and
          // starts returning ERROR_IMAGE_BUFFER_FULL instead of frames if they
          // aren't returned.
          image.delete()
        }
      }

      if (!module.isOk(result.error)) {
        // Not an error so much as "that's all for now" -- the normal way a
        // decode call ends when the next frame needs more input.
        if (result.error === module.Error.ERROR_WAITING_FOR_INPUT_DATA) return
        throw new Error(`HEVC decoding failed: ${module.getErrorText(result.error)}`)
      }
    }
  }

  private toVideoSample(image: ReturnType<Decoder['getNextPicture']> & {}): VideoSample {
    if (image.chromaFormat !== CHROMA_420) {
      throw new Error(`This HEVC file uses a chroma format this app can't handle (${image.chromaFormat}); action-cam footage is normally 4:2:0`)
    }

    const bitsPerSample = image.getBitsPerPixel(0)
    const format: VideoSamplePixelFormat = bitsPerSample > 8 ? 'I420P10' : 'I420'
    const width = image.getWidth(0)
    const height = image.getHeight(0)

    // libde265 hands back three separately-allocated planes, while VideoSample
    // wants one buffer described by per-plane offsets -- so they're packed
    // together here. The copy is needed regardless: these are views into the
    // wasm heap and are invalidated by `image.delete()` below.
    const planes = [0, 1, 2].map((channel) => image.getImagePlane(channel))
    const layout = []
    let total = 0
    for (const plane of planes) {
      layout.push({ offset: total, stride: plane.stride })
      total += plane.stride * plane.height
    }
    const data = new Uint8Array(total)
    for (const [i, plane] of planes.entries()) data.set(plane.bytes.subarray(0, plane.stride * plane.height), layout[i].offset)

    return new VideoSample(data, {
      format,
      codedWidth: width,
      codedHeight: height,
      // No `visibleRect`: libde265 has already applied the conformance window,
      // so a stream coded as 1088 tall to fill whole CTUs is handed back at
      // its real 1080. Verified against ffmpeg, which produces the same
      // cropped frame byte for byte.
      timestamp: Number(image.pts) / MICROSECONDS_PER_SECOND,
      layout,
      colorSpace: {
        fullRange: image.isFullRange,
        primaries: PRIMARIES[image.colorPrimaries],
        transfer: TRANSFER[image.transferCharacteristics],
        matrix: MATRIX[image.matrixCoefficients],
      } as VideoColorSpaceInit,
    })
  }

  async close(): Promise<void> {
    // The wasm module itself is deliberately left loaded and cached; only this
    // decoder's own state is released.
    for (const sample of this.pending) sample.close()
    this.pending = []
    this.decoder?.delete()
    this.decoder = null
  }
}
