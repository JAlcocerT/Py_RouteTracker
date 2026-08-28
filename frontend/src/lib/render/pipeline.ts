/**
 * The client-side replacement for backend/app/render/video_render.py's
 * `render_and_composite`: trim + draw the HUD + composite it onto the real
 * footage, in one pass. Where the server version split this into a cheap
 * "trim" phase (ffmpeg re-encode of the whole requested range) and an
 * expensive "render" phase (parallel matplotlib frame rendering + a second
 * ffmpeg overlay pass), this collapses both into a single decode -> draw ->
 * encode pipeline via mediabunny's `Conversion`, which already knows how to
 * trim, decode, and passthrough-copy audio without a separate re-encode
 * step. The whole thing never touches the source video outside
 * [trimStart, trimEnd] at all -- an improvement over the server pipeline,
 * which always fully re-encoded the trim range as a distinct step first.
 *
 * Decoding normally runs on the browser's own (hardware) decoders via
 * WebCodecs. HEVC is the exception -- most browser builds can't decode it at
 * all -- so hevcDecoder.ts registers a WASM decoder with mediabunny for those
 * files. That swap is invisible here: it replaces one step inside
 * `Conversion` and leaves trim, the HUD draw, muxing, and the encode alone.
 *
 * Not validated end-to-end in a real browser against real footage (see this
 * repo's other render-pipeline modules for the same caveat) -- in
 * particular, `VideoSample.timestamp`'s exact meaning inside `Conversion`'s
 * `process` callback (absolute to the untrimmed input vs. rebased to the
 * trim window) is assumed, not confirmed, to be the former; getting this
 * wrong would misalign the HUD against the footage without erroring.
 */
import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, StreamTarget, WebMOutputFormat } from 'mediabunny'
import type { OutputContainer } from '../env'
import type { DiscardedTrack } from 'mediabunny'
import { hasWebCodecsSupport, isInsecureContext } from '../env'
import type { AnnotatedRow } from '../laps/detection'
import { ensureSoftwareHevcDecoder } from './hevcDecoder'
import { HudRenderer } from './hudRenderer'
import type { RenderConfig } from './renderConfig'

interface DiscardedTrackDiagnostic {
  type: string
  number: number
  codec: string
  reason: string
  configSummary: string
}

/** Captures exactly what config each discarded track's real decode check was
 * built from -- the file's actual codec/description/resolution/sample rate,
 * not the generic guessed config `checkActionCamCodecSupport` (lib/env.ts)
 * probes with -- so a real failure report carries the actual cause instead
 * of another guess (the previous message here guessed "licensed codec
 * support gap", which didn't hold up against a real report of AAC failing
 * on an official Chrome install). Never throws: test doubles for
 * `Conversion` don't implement the full InputTrack interface, and a
 * diagnostics failure shouldn't take down the real error path either way --
 * both fall back to "no decoder config available" via the catch below. */
async function describeDiscardedTracks(discardedTracks: readonly DiscardedTrack[]): Promise<DiscardedTrackDiagnostic[]> {
  return Promise.all(
    discardedTracks.map(async (d) => {
      const track = d.track
      let configSummary = 'no decoder config available'
      try {
        if (track.isVideoTrack()) {
          configSummary = summarizeDecoderConfig(await track.getDecoderConfig())
        } else if (track.isAudioTrack()) {
          configSummary = summarizeDecoderConfig(await track.getDecoderConfig())
        }
      } catch {
        // best-effort, see docstring above
      }
      return { type: track.type, number: track.number, codec: track.codec ?? 'unknown', reason: d.reason, configSummary }
    }),
  )
}

function summarizeDecoderConfig(config: VideoDecoderConfig | AudioDecoderConfig | null): string {
  if (!config) return 'no decoder config available'
  const { description, ...rest } = config
  const descriptionBytes = description == null ? 0 : ArrayBuffer.isView(description) ? description.byteLength : (description as ArrayBuffer).byteLength
  return JSON.stringify({ ...rest, descriptionBytes })
}

export interface RenderVideoOptions {
  trimStart: number
  trimEnd: number
  config: Omit<RenderConfig, 'widthPx' | 'heightPx'>
  /** Full-session, lap-annotated telemetry (see app.laps.detection.detectLaps)
   * -- windowed to [trimStart, trimEnd] internally. Lap/chrono fields are
   * already correct relative to the full session and are not recomputed. */
  annotatedRows: AnnotatedRow[]
  onProgress?: (fraction: number) => void
  /** Reports whether this render is using the software HEVC decoder, so the
   * UI can explain why a render is slow rather than leaving a progress bar to
   * crawl with no explanation. */
  onStage?: (stage: RenderStage) => void
  /** When given, output is streamed directly to this sink (e.g. a
   * FileSystemWritableFileStream from window.showSaveFilePicker) instead of
   * being buffered entirely in memory. */
  outputStream?: WritableStream
  /** Chosen by the caller via `pickOutputContainer` before the save-file
   * picker opens, so the suggested filename matches what's written. */
  outputContainer?: OutputContainer
}

export type RenderStage = 'rendering' | 'software-decoding'

export async function renderVideo(videoFile: File, options: RenderVideoOptions): Promise<Blob | null> {
  const { trimStart, trimEnd, config, annotatedRows, onProgress, onStage, outputStream, outputContainer = 'mp4' } = options
  const windowedRows = annotatedRows.filter((r) => r.time >= trimStart && r.time <= trimEnd)
  if (windowedRows.length === 0) throw new Error('No telemetry samples in the selected trim range')

  const input = new Input({ source: new BlobSource(videoFile), formats: ALL_FORMATS })
  const videoTrack = await input.getPrimaryVideoTrack()
  if (!videoTrack) throw new Error('No video track found in the source file')

  // The HUD canvas must be pixel-for-pixel the same size as the real
  // (display-oriented) footage -- see hud_layers.py's config_for_resolution
  // docstring for why an unscaled, corner-anchored overlay needs this.
  const width = await videoTrack.getDisplayWidth()
  const height = await videoTrack.getDisplayHeight()
  const hud = new HudRenderer(windowedRows, { ...config, widthPx: width, heightPx: height })

  // Registers the WASM HEVC decoder with mediabunny if -- and only if -- this
  // browser can't decode this particular file natively. Everything below is
  // unaffected either way: `Conversion` picks the registered decoder up on its
  // own, and the encode still runs on the browser's own encoder. See
  // hevcDecoder.ts for why so much footage needs this.
  const usingSoftwareDecode = await ensureSoftwareHevcDecoder(await videoTrack.getDecoderConfig())
  if (usingSoftwareDecode) onStage?.('software-decoding')

  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not acquire a 2D canvas context')

  const target = outputStream ? new StreamTarget(outputStream as WritableStream<never>) : new BufferTarget()
  const output = new Output({ format: outputContainer === 'webm' ? new WebMOutputFormat() : new Mp4OutputFormat(), target })

  const conversion = await Conversion.init({
    input,
    output,
    // Without this, Conversion also considers every other input track --
    // notably the GoPro file's own 'gpmd' GPS/telemetry track, which has no
    // codec Conversion recognizes. That track being discarded is harmless on
    // its own, but mixed in with a real discard reason (e.g. the browser
    // being unable to decode/encode the video codec) it was muddying
    // `discardedTracks` and the resulting error. Source footage only ever
    // has one video + optionally one audio track worth keeping anyway.
    tracks: 'primary',
    trim: { start: trimStart, end: trimEnd },
    video: {
      process: (sample) => {
        sample.draw(ctx, 0, 0, width, height)
        hud.drawFrameAtTime(ctx, sample.timestamp)
        return canvas
      },
    },
  })

  if (!conversion.isValid) {
    // The generic per-codec 'undecodable_source_codec' message guessed at a
    // cause (licensed-codec-support gap) that turned out not to explain a
    // real report -- an official Chrome install with AAC failing alongside
    // HEVC, which shouldn't happen since Chrome bundles AAC decode
    // everywhere. Rather than guess again, capture exactly what config each
    // track's real decoder check was built from (not the generic guessed
    // config `checkActionCamCodecSupport` in lib/env.ts probes with) so the
    // next report carries the actual cause instead of another theory.
    const diagnostics = await describeDiscardedTracks(conversion.discardedTracks)
    // eslint-disable-next-line no-console
    console.error(
      '[renderVideo] Conversion invalid -- discarded tracks:',
      diagnostics,
      'userAgent:',
      typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
    )

    if (!hasWebCodecsSupport()) {
      // The whole API is missing, not just support for this file's codecs --
      // every track gets discarded as 'undecodable_source_codec' regardless
      // of what codec it actually is, which reads exactly like a codec
      // problem below. Name the real cause instead. Note the software HEVC
      // decoder can't stand in here: it only replaces the *decode* step, and
      // encoding still needs WebCodecs, so with the API absent entirely there
      // is nothing to fall back to.
      throw new Error(
        isInsecureContext()
          ? "This page can't decode or encode video because it's loaded over an insecure connection -- WebCodecs (which rendering depends on) is only available over HTTPS or from localhost. Open this app via HTTPS, or over localhost/127.0.0.1, instead."
          : "This browser doesn't support the WebCodecs APIs this app needs to decode and encode video. Try a recent Chrome, Edge, or other Chromium-based browser.",
      )
    }

    const detail = diagnostics.map((d) => `${d.type} track #${d.number} (codec: ${d.codec}): ${d.reason} [${d.configSummary}]`).join('; ')
    throw new Error(
      detail
        ? `Your browser can't render this video: ${detail} -- open the browser console for the full decoder config (helps diagnose exactly why).`
        : "Your browser can't render this video for an unknown reason -- no tracks could be read from it.",
    )
  }

  conversion.onProgress = (fraction) => onProgress?.(fraction)
  await conversion.execute()

  if (outputStream) return null
  return new Blob([(target as BufferTarget).buffer!], { type: outputContainer === 'webm' ? 'video/webm' : 'video/mp4' })
}
