/** Reads a video File's duration client-side -- replaces the server's
 * ffprobe-based `get_video_duration` (app/core/ffmpeg_utils.py).
 *
 * Two strategies, in order. A throwaway <video> element is tried first
 * because it's instant and needs no container parsing. But it only works
 * for codecs the browser can actually play, and this app's whole HEVC
 * problem is that plenty of browser builds can't (Chrome on Linux gates
 * HEVC behind VAAPI; distro Chromium ships no proprietary codecs at all).
 * A hard reject there used to fail the upload outright, before the user
 * ever reached the trim/render steps.
 *
 * So the fallback reads the duration out of the container via mediabunny,
 * which parses boxes/packets and never instantiates a decoder -- meaning it
 * works regardless of whether this browser can decode the video. */
import { ALL_FORMATS, BlobSource, Input } from 'mediabunny'

const VIDEO_METADATA_TIMEOUT_MS = 10_000

function durationFromVideoElement(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    let settled = false

    // A <video> that can't handle the codec sometimes fires neither
    // loadedmetadata nor error -- without this timeout the whole upload
    // would hang on it rather than falling through to the container read.
    const timer = setTimeout(() => finish(() => reject(new Error('timed out reading video metadata'))), VIDEO_METADATA_TIMEOUT_MS)

    function finish(settle: () => void) {
      if (settled) return
      settled = true
      clearTimeout(timer)
      URL.revokeObjectURL(url)
      settle()
    }

    video.preload = 'metadata'
    video.onloadedmetadata = () => {
      const duration = video.duration
      finish(() =>
        Number.isFinite(duration) && duration > 0
          ? resolve(duration)
          : reject(new Error('video element reported no usable duration')),
      )
    }
    video.onerror = () => finish(() => reject(new Error('video element could not read this file')))
    video.src = url
  })
}

async function durationFromContainer(file: File): Promise<number> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  // Metadata first (cheap header read); computeDuration walks packets and is
  // slower, so it's only the last resort for containers with no stated duration.
  const fromMetadata = await input.getDurationFromMetadata()
  if (fromMetadata != null && Number.isFinite(fromMetadata) && fromMetadata > 0) return fromMetadata

  const computed = await input.computeDuration()
  if (!Number.isFinite(computed) || computed <= 0) throw new Error('container reported no usable duration')
  return computed
}

export async function probeVideoDuration(file: File): Promise<number> {
  try {
    return await durationFromVideoElement(file)
  } catch {
    // Expected whenever the browser can't play this codec -- fall through.
  }

  try {
    return await durationFromContainer(file)
  } catch (error) {
    throw new Error(
      `Could not read video metadata for '${file.name}': ${error instanceof Error ? error.message : String(error)}`,
    )
  }
}

/** Whether this browser's <video> element can actually play the file, as
 * opposed to merely being able to read its duration. Drives the trimmer's
 * preview: `false` means show the telemetry-only trimmer rather than a
 * permanently-black player (see components/VideoTrimmer.tsx). */
export async function canPreviewVideo(file: File): Promise<boolean> {
  try {
    await durationFromVideoElement(file)
    return true
  } catch {
    return false
  }
}
