/**
 * Software-decode fallback for footage this browser's own decoders can't
 * handle. See pipeline.ts for where it's triggered.
 *
 * The render pipeline normally decodes via WebCodecs, which hands the work
 * to the browser's built-in (usually hardware-accelerated) decoders. That's
 * fast, but it can only expose codecs the browser actually ships:
 *
 *   - Chrome/Edge decode HEVC only through hardware. On Linux that means
 *     VAAPI -- realistically Intel GPUs -- so NVIDIA/AMD desktops, VMs and
 *     headless boxes get nothing, with no software decoder to fall back on.
 *   - Chromium builds packaged by Linux distros are frequently compiled
 *     without the patent-licensed codecs entirely, losing H.264, HEVC *and*
 *     AAC together.
 *
 * Either way the fix is the same: bring our own decoder. ffmpeg.wasm carries
 * its own HEVC/H.264/AAC implementations, so it doesn't care what the
 * browser shipped.
 *
 * The output intermediate is deliberately VP8 + Opus in WebM, not H.264/AAC:
 * a browser that reached this fallback may well be one with *no* licensed
 * codecs at all, so re-encoding to H.264 would land right back on an
 * undecodable file. VP8 and Opus are royalty-free and present in every
 * browser build, which is exactly the property needed here.
 */
import type { FFmpeg } from '@ffmpeg/ffmpeg'

// Self-hosted rather than loaded from a CDN: this app is an offline-capable
// PWA that makes a point of never phoning home (see README's privacy
// section), and a CDN fetch would break both properties. Vite emits these as
// regular build assets; the ~32MB .wasm is far above workbox's default
// precache size cap, so the service worker leaves it to be fetched on demand
// -- which is what we want, since most users never hit this path.
// The ESM build specifically: @ffmpeg/ffmpeg spawns its internal worker with
// `type: "module"`, where `importScripts` doesn't exist, so it loads the core
// via dynamic `import()` and needs the ESM entry rather than the UMD one.
import coreURL from '@ffmpeg/core?url'
import wasmURL from '@ffmpeg/core/wasm?url'

/** VP8 rather than VP9: this path is already the slow one, and libvpx's VP9
 * encoder is dramatically slower in WASM for a quality gain that a
 * throwaway intermediate doesn't benefit from. The high bitrate keeps this
 * generation from visibly degrading the footage before the real encode. */
const INTERMEDIATE_VIDEO_BITRATE = '12M'
const INTERMEDIATE_AUDIO_BITRATE = '128k'
const MOUNT_POINT = '/mnt'
const OUTPUT_NAME = 'transcoded.webm'

let ffmpegPromise: Promise<FFmpeg> | null = null

/** Loaded once per worker and reused -- the core is ~32MB, so a second
 * render in the same session shouldn't pay for it again. */
async function loadFFmpeg(onLog?: (line: string) => void): Promise<FFmpeg> {
  if (!ffmpegPromise) {
    ffmpegPromise = (async () => {
      const { FFmpeg } = await import('@ffmpeg/ffmpeg')
      const ffmpeg = new FFmpeg()
      if (onLog) ffmpeg.on('log', ({ message }) => onLog(message))
      await ffmpeg.load({ coreURL, wasmURL })
      return ffmpeg
    })()
    // A failed load must not be cached, or every later attempt in this
    // session resolves to the same rejection with no way to retry.
    ffmpegPromise.catch(() => {
      ffmpegPromise = null
    })
  }
  return ffmpegPromise
}

export interface WasmTranscodeOptions {
  /** Only this window is transcoded. Software decode is slow enough that
   * transcoding a whole 30-minute recording to render a 2-minute cut from
   * it would be a serious regression -- and the caller has already asked
   * for a trim, so the rest is work nobody wants done. */
  trimStart: number
  trimEnd: number
  onProgress?: (fraction: number) => void
  onLog?: (line: string) => void
}

/**
 * Transcodes `[trimStart, trimEnd]` of `file` into a WebM this browser is
 * guaranteed to be able to decode. The returned file's timeline starts at
 * zero, so callers must rebase their own timestamps by `-trimStart` -- see
 * pipeline.ts, which shifts the telemetry rows to match.
 */
export async function transcodeForCompatibility(file: File, options: WasmTranscodeOptions): Promise<File> {
  const { trimStart, trimEnd, onProgress, onLog } = options
  const ffmpeg = await loadFFmpeg(onLog)

  const progressHandler = ({ progress }: { progress: number }) => {
    // ffmpeg reports progress against the *trimmed* duration, and can
    // briefly overshoot 1 near the end.
    onProgress?.(Math.min(Math.max(progress, 0), 1))
  }
  ffmpeg.on('progress', progressHandler)

  // WORKERFS exposes the File to ffmpeg as a real file backed by the
  // original Blob, rather than copying its bytes into the WASM heap the way
  // writeFile() does. Action-cam footage routinely runs to several GB, well
  // past the 4GB ceiling a 32-bit WASM heap can address at all, so this is
  // load-bearing rather than merely an optimization.
  await ffmpeg.createDir(MOUNT_POINT)
  await ffmpeg.mount('WORKERFS' as Parameters<FFmpeg['mount']>[0], { files: [file] }, MOUNT_POINT)

  try {
    await ffmpeg.exec([
      // Before -i, so ffmpeg seeks rather than decoding-and-discarding
      // everything up to trimStart.
      '-ss',
      String(trimStart),
      '-to',
      String(trimEnd),
      '-i',
      `${MOUNT_POINT}/${file.name}`,
      '-c:v',
      'libvpx',
      '-b:v',
      INTERMEDIATE_VIDEO_BITRATE,
      // Favour throughput over compression: this file is decoded once and
      // thrown away, so spending CPU to shrink it is wasted on the one path
      // where CPU is already the bottleneck.
      '-deadline',
      'realtime',
      '-cpu-used',
      '8',
      '-c:a',
      'libopus',
      '-b:a',
      INTERMEDIATE_AUDIO_BITRATE,
      OUTPUT_NAME,
    ])

    const data = await ffmpeg.readFile(OUTPUT_NAME)
    // readFile returns string only when asked for utf8; binary reads give a
    // Uint8Array. Guard anyway so a surprise here fails loudly rather than
    // producing a corrupt zero-length video.
    if (typeof data === 'string') throw new Error('ffmpeg returned text where video data was expected')
    return new File([data as BlobPart], `${file.name.replace(/\.[^.]+$/, '')}_compat.webm`, { type: 'video/webm' })
  } finally {
    ffmpeg.off('progress', progressHandler)
    await ffmpeg.deleteFile(OUTPUT_NAME).catch(() => undefined)
    await ffmpeg.unmount(MOUNT_POINT).catch(() => undefined)
    await ffmpeg.deleteDir(MOUNT_POINT).catch(() => undefined)
  }
}
