/**
 * Lossless joining of split action-cam recordings. Ported from
 * backend/app/core/video_join.py, which used ffmpeg's concat demuxer to
 * stream-copy parts together (no re-encode, every stream -- including a
 * GoPro's `gpmd` telemetry track -- preserved verbatim at its original
 * index). This does the same thing at the box level with mp4box.js instead
 * of shelling out to ffmpeg: demux every track's raw samples from every
 * part, then author one fresh output file with mp4box.js's track-authoring
 * API (`addTrack`/`addSample`), copying each track's original sample
 * description box (`Sample.description`, whatever codec or -- for `gpmd`,
 * whose format mp4box.js has no built-in knowledge of -- unrecognized box
 * type it is) straight from the source into the new track, byte for byte.
 *
 * Caveats this module hasn't been able to verify against a real multi-part
 * GoPro recording (no such fixture exists in this repo):
 *  - Whether mp4box.js's generic box parser round-trips an unrecognized
 *    `gpmd` sample entry cleanly enough to reuse as another track's
 *    description (ffmpeg needed an explicit `-copy_unknown` flag for the
 *    equivalent; mp4box.js has no documented equivalent to confirm against).
 *  - mp4box.js's authoring API builds the whole output (moov + mdat) in
 *    memory -- there is no streaming-to-disk output path the way the
 *    demux side can stream from a File. For the multi-GB joined files this
 *    feature exists for, that's a real memory ceiling, not just a
 *    theoretical one.
 * This needs a manual test pass against real chaptered footage before
 * being trusted the way the ffmpeg version was.
 */
import { createFile, type Box, type ISOFile, type Movie, type MP4BoxBuffer, type Sample, type Track } from 'mp4box'

export class IncompatiblePartsError extends Error {}

interface TrackSignature {
  codec: string
  type: 'video' | 'audio' | 'metadata'
  width: number
  height: number
  timescale: number
}

interface DemuxedTrack {
  info: Track
  samples: Sample[]
}

interface DemuxedFile {
  tracks: DemuxedTrack[]
}

function trackKind(track: Track): 'video' | 'audio' | 'metadata' {
  if (track.codec === 'gpmd') return 'metadata'
  if (track.type === 'video' || track.video) return 'video'
  if (track.type === 'audio' || track.audio) return 'audio'
  return 'metadata'
}

function signatureOf(track: Track): TrackSignature {
  return {
    codec: track.codec,
    type: trackKind(track),
    width: track.video?.width ?? 0,
    height: track.video?.height ?? 0,
    timescale: track.timescale,
  }
}

/** Streams `file` through mp4box.js in chunks (never holding the whole file
 * in memory at once) and collects every track's raw samples. */
async function demuxFile(file: File, onProgress?: (fraction: number) => void): Promise<DemuxedFile> {
  return new Promise((resolve, reject) => {
    const mp4boxFile = createFile()
    const collected = new Map<number, DemuxedTrack>()

    mp4boxFile.onError = (error: string) => reject(new Error(error))

    mp4boxFile.onReady = (info: Movie) => {
      for (const track of info.tracks) {
        collected.set(track.id, { info: track, samples: [] })
        mp4boxFile.setExtractionOptions(track.id, undefined, { nbSamples: track.nb_samples })
      }
      mp4boxFile.start()
    }

    mp4boxFile.onSamples = (id: number, _user: unknown, samples: Sample[]) => {
      collected.get(id)?.samples.push(...samples)
    }

    const fileSize = file.size
    let offset = 0
    const reader = file.stream().getReader()

    function pump(): void {
      reader
        .read()
        .then(({ done, value }) => {
          if (done) {
            mp4boxFile.flush()
            resolve({ tracks: Array.from(collected.values()) })
            return
          }
          const buffer = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength) as MP4BoxBuffer
          buffer.fileStart = offset
          offset += value.byteLength
          mp4boxFile.appendBuffer(buffer)
          onProgress?.(Math.min(1, offset / fileSize))
          pump()
        })
        .catch(reject)
    }
    pump()
  })
}

/** Raises IncompatiblePartsError if the given parts don't share the same
 * track layout as the first part -- true for chaptered recordings from the
 * same camera session, not for unrelated clips someone selected together. */
export async function validateJoinCompatible(parts: File[]): Promise<DemuxedFile[]> {
  if (parts.length < 2) throw new IncompatiblePartsError('need at least two video parts to join')

  const demuxed = await Promise.all(parts.map((p) => demuxFile(p)))
  const signaturesOf = (d: DemuxedFile) => d.tracks.map((t) => signatureOf(t.info))
  const first = signaturesOf(demuxed[0])

  for (let i = 1; i < demuxed.length; i++) {
    const sig = signaturesOf(demuxed[i])
    const mismatch =
      sig.length !== first.length || sig.some((s, k) => JSON.stringify(s) !== JSON.stringify(first[k]))
    if (mismatch) {
      throw new IncompatiblePartsError(
        `'${parts[i].name}' doesn't match the first part's format -- a lossless join needs every part ` +
          'to share the same tracks, codecs, resolution, and frame rate. This usually means these files ' +
          "aren't chapters of the same recording, or one of them has been re-encoded.",
      )
    }
  }
  return demuxed
}

/** Losslessly concatenates already-demuxed, validated-compatible parts into
 * one MP4 Blob, preserving every track (including a GoPro's gpmd
 * telemetry track) at its original sample description. */
function authorJoinedFile(demuxedParts: DemuxedFile[]): Blob {
  const out: ISOFile = createFile()

  // Map first-part track id -> output track id, so later parts' samples
  // (matched positionally, since validateJoinCompatible already confirmed
  // identical track layouts) land on the track their own kind belongs to.
  const outputTrackIds: number[] = []
  const ctsOffsetByIndex: number[] = []

  const firstPart = demuxedParts[0]
  for (const { info, samples } of firstPart.tracks) {
    // No `type` option: the original, fully-formed sample description box
    // (`description`, straight from the source) is what actually determines
    // the track's real codec/box-type on write -- passing it directly
    // avoids needing to map every possible codec string onto mp4box.js's
    // fixed SampleEntryFourCC registry, which doesn't have an entry for
    // `gpmd` at all.
    const outId = out.addTrack({
      width: info.video?.width,
      height: info.video?.height,
      timescale: info.timescale,
      samplerate: info.audio?.sample_rate,
      channel_count: info.audio?.channel_count,
      samplesize: info.audio?.sample_size,
      // `Sample.description` is typed as `SampleGroupEntry | SampleEntry`
      // (the description column can, in principle, hold either) -- for an
      // actual media sample it's always the SampleEntry (the per-track
      // codec/box description), which is itself a Box.
      description: samples[0]?.description as Box | undefined,
    })
    outputTrackIds.push(outId)
    ctsOffsetByIndex.push(0)
  }

  for (const part of demuxedParts) {
    part.tracks.forEach((track, i) => {
      const outId = outputTrackIds[i]
      let offset = ctsOffsetByIndex[i]
      let maxEnd = offset
      for (const sample of track.samples) {
        out.addSample(outId, sample.data!, {
          duration: sample.duration,
          cts: sample.cts + offset,
          dts: sample.dts + offset,
          is_sync: sample.is_sync,
        })
        maxEnd = Math.max(maxEnd, sample.cts + offset + sample.duration)
      }
      ctsOffsetByIndex[i] = maxEnd
    })
  }

  return out.save('joined.mp4')
}

export async function joinVideos(parts: File[], onProgress?: (fraction: number) => void): Promise<Blob> {
  const demuxedParts = await validateJoinCompatible(parts)
  onProgress?.(1)
  return authorJoinedFile(demuxedParts)
}
