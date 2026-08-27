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
 * Not validated end-to-end in a real browser against real footage (see this
 * repo's other render-pipeline modules for the same caveat) -- in
 * particular, `VideoSample.timestamp`'s exact meaning inside `Conversion`'s
 * `process` callback (absolute to the untrimmed input vs. rebased to the
 * trim window) is assumed, not confirmed, to be the former; getting this
 * wrong would misalign the HUD against the footage without erroring.
 */
import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, StreamTarget } from 'mediabunny'
import { hasWebCodecsSupport, isInsecureContext } from '../env'
import type { AnnotatedRow } from '../laps/detection'
import { HudRenderer } from './hudRenderer'
import type { RenderConfig } from './renderConfig'

export interface RenderVideoOptions {
  trimStart: number
  trimEnd: number
  config: Omit<RenderConfig, 'widthPx' | 'heightPx'>
  /** Full-session, lap-annotated telemetry (see app.laps.detection.detectLaps)
   * -- windowed to [trimStart, trimEnd] internally. Lap/chrono fields are
   * already correct relative to the full session and are not recomputed. */
  annotatedRows: AnnotatedRow[]
  onProgress?: (fraction: number) => void
  /** When given, output is streamed directly to this sink (e.g. a
   * FileSystemWritableFileStream from window.showSaveFilePicker) instead of
   * being buffered entirely in memory. */
  outputStream?: WritableStream
}

export async function renderVideo(videoFile: File, options: RenderVideoOptions): Promise<Blob | null> {
  const { trimStart, trimEnd, config, annotatedRows, onProgress, outputStream } = options
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

  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not acquire a 2D canvas context')

  const target = outputStream ? new StreamTarget(outputStream as WritableStream<never>) : new BufferTarget()
  const output = new Output({ format: new Mp4OutputFormat(), target })

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
    // A whole-page VideoDecoder/AudioDecoder outage (not a real per-codec
    // gap) shows up here as every track being discarded as
    // 'undecodable_source_codec' regardless of what codec it actually is --
    // catch that case with a message that names the real cause instead of
    // one that reads like this specific video's codec isn't supported.
    const allUndecodable = conversion.discardedTracks.length > 0 && conversion.discardedTracks.every((d) => d.reason === 'undecodable_source_codec')
    if (allUndecodable && !hasWebCodecsSupport()) {
      throw new Error(
        isInsecureContext()
          ? "This page can't decode or encode video because it's loaded over an insecure connection -- WebCodecs (which rendering depends on) is only available over HTTPS or from localhost. Open this app via HTTPS, or over localhost/127.0.0.1, instead."
          : "This browser doesn't support the WebCodecs APIs this app needs to decode and encode video. Try a recent Chrome, Edge, or other Chromium-based browser.",
      )
    }

    const reasons = conversion.discardedTracks
      .map((d) => `${d.track.type} track #${d.track.number} (codec: ${d.track.codec ?? 'unknown'}): ${d.reason}`)
      .join('; ')
    throw new Error(
      reasons
        ? `Your browser can't render this video: ${reasons}`
        : "Your browser can't render this video for an unknown reason -- no tracks could be read from it.",
    )
  }

  conversion.onProgress = (fraction) => onProgress?.(fraction)
  await conversion.execute()

  if (outputStream) return null
  return new Blob([(target as BufferTarget).buffer!], { type: 'video/mp4' })
}
