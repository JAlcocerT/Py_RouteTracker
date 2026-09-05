/**
 * Lossless joining of split action-cam recordings. Ported from
 * backend/app/core/video_join.py, which used ffmpeg's concat demuxer to
 * stream-copy parts together (no re-encode, every stream -- including a
 * GoPro's `gpmd` telemetry track -- preserved verbatim at its original
 * index). This does the same thing at the box level with mp4box.js instead
 * of shelling out to ffmpeg: demux every track's raw samples from every
 * part, then author one fresh output file with mp4box.js's track-authoring
 * API (`addTrack`/`addSample`), reusing each source track's own sample
 * description box (its `stsd` entry) in the output byte for byte.
 *
 * On reusing the source `stsd` entry: mp4box.js's `addTrack` does *not*
 * take a ready-made sample entry. It always constructs one from its
 * `options.type` fourcc (defaulting to `avc1` when omitted) and treats
 * `options.description` as a *child* box to nest inside it. Handing it the
 * source's `hvc1`/`gpmd`/`mp4a` entry that way produced an output whose
 * every track was an `avc1` video track with the real entry buried inside
 * -- HEVC footage mislabelled as AVC, and the `gpmd` telemetry track
 * (which has no entry in mp4box.js's SampleEntryFourCC registry at all)
 * destroyed outright, which is what `extractGoProGpmf` then found nothing
 * in. So instead `addTrack` is called with a *registered placeholder*
 * fourcc of the right media kind purely to get a well-formed `trak`
 * skeleton, and the placeholder `stsd` entry and handler are then replaced
 * with the source's own. mp4box.js keeps the raw bytes of sample entries it
 * can't parse (see `stsdBox.parse`'s `parseDataAndRewind` fallback), so an
 * unrecognized `gpmd` entry round-trips through that swap unchanged.
 *
 * Memory: mp4box.js's authoring API builds the whole output (moov + mdat)
 * in memory -- there is no streaming-to-disk output path the way the demux
 * side can stream from a File. Parts are therefore demuxed one at a time
 * (not all at once) and mp4box.js's own copy of each sample is released as
 * soon as it has been collected, but peak usage is still bounded below by
 * the size of the joined file. mediabunny, which this app uses for
 * streaming muxing everywhere else, is not an option here: it only models
 * video/audio/subtitle tracks (`ALL_TRACK_TYPES`) and would silently drop
 * the telemetry track this whole feature exists to preserve.
 *
 * Still unverified against a real multi-part GoPro recording (no such
 * fixture exists in this repo): whether a real camera's `gpmd` sample entry
 * survives the round trip in practice, the way ffmpeg needed an explicit
 * `-copy_unknown` for the equivalent.
 */
import { createFile, type ISOFile, type Movie, type MP4BoxBuffer, type Sample, type SampleEntry, type Track } from 'mp4box'

export class IncompatiblePartsError extends Error {}

/** Movie-level timescale for the authored output. Only has to be fine
 * enough to express the total duration; each track keeps its own. */
const MOVIE_TIMESCALE = 1000

/** Fraction of the reported progress spent demuxing the parts; the rest is
 * the authoring pass. Demuxing reads and parses every byte of every part
 * while authoring only re-serializes samples already in memory, so the
 * split is deliberately lopsided. */
const DEMUX_PROGRESS_SHARE = 0.85

/** How many samples to author between yields to the event loop, so a
 * multi-hundred-thousand-sample join still reports progress as it goes
 * rather than blocking straight through to the end. */
const AUTHOR_YIELD_INTERVAL = 20_000

/** A real macrotask yield. `await Promise.resolve()` would not do: a
 * microtask runs before the event loop gets a turn at all, so the host would
 * still be unable to deliver the progress that was just reported. */
const yieldToEventLoop = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

type TrackKind = 'video' | 'audio' | 'metadata'

/** Registered fourccs used only to make `addTrack` build a `trak` of the
 * right shape (the matching media header box -- `vmhd`/`smhd`/`nmhd`) before
 * the real sample entry is swapped in over the top. Anything not video or
 * audio goes through the metadata placeholder; the source's own handler is
 * restored afterwards either way, so a subtitle track keeps its `sbtl`
 * handler even though it borrows the metadata skeleton. All three are types
 * mp4box.js can actually *write*, not merely construct -- several of its
 * metadata entries (`mett`, `metx`, `tx3g`) implement only `parse` and throw
 * on save, which would matter in the edge case of a source track that has no
 * sample entry to swap in over the placeholder. */
const PLACEHOLDER_FOURCC = { video: 'avc1', audio: 'mp4a', metadata: 'mp4s' } as const

interface TrackSignature {
  codec: string
  type: TrackKind
  width: number
  height: number
  timescale: number
}

/** The subset of an mp4box.js `Sample` this module needs, copied out so
 * that mp4box.js's own sample store can be released as we go. `data` is
 * already a private per-sample copy inside mp4box.js (`getSample` allocates
 * it), not a view into the appended buffers, so taking the reference and
 * letting mp4box.js forget about it hands ownership over cleanly. */
interface JoinSample {
  data: Uint8Array<ArrayBuffer>
  duration: number
  cts: number
  dts: number
  is_sync: boolean
}

interface DemuxedTrack {
  info: Track
  /** The source track's `hdlr` handler type (`vide`/`soun`/`meta`/...), kept
   * verbatim -- it is what identifies a GoPro telemetry track downstream. */
  handler: string
  /** The source track's own `stsd` entry, reused as-is in the output. */
  description: SampleEntry | undefined
  samples: JoinSample[]
}

interface DemuxedFile {
  tracks: DemuxedTrack[]
}

function trackKind(handler: string, track: Track): TrackKind {
  if (handler === 'vide') return 'video'
  if (handler === 'soun') return 'audio'
  if (handler) return 'metadata'
  // No handler to go on (malformed source): fall back to what the parsed
  // track info claims.
  if (track.type === 'video' || track.video) return 'video'
  if (track.type === 'audio' || track.audio) return 'audio'
  return 'metadata'
}

function signatureOf(track: DemuxedTrack): TrackSignature {
  return {
    codec: track.info.codec,
    type: trackKind(track.handler, track.info),
    width: track.info.video?.width ?? 0,
    height: track.info.video?.height ?? 0,
    timescale: track.info.timescale,
  }
}

/** Streams `file` through mp4box.js in chunks (never holding the whole file
 * in memory at once) and collects every track's raw samples. */
async function demuxFile(file: File, onProgress?: (fraction: number) => void): Promise<DemuxedFile> {
  return new Promise((resolve, reject) => {
    const mp4boxFile = createFile()
    const collected = new Map<number, DemuxedTrack>()

    // mp4box.js calls this with (module, message), not just a message --
    // taking only the first argument reported the module name ('ISOFile')
    // as the whole error.
    mp4boxFile.onError = (module: string, message: string) => reject(new Error(`${module}: ${message}`))

    mp4boxFile.onReady = (info: Movie) => {
      for (const track of info.tracks) {
        const trak = mp4boxFile.getTrackById(track.id)
        collected.set(track.id, {
          info: track,
          handler: trak?.mdia.hdlr.handler ?? '',
          description: trak?.mdia.minf.stbl.stsd.entries[0],
          samples: [],
        })
        mp4boxFile.setExtractionOptions(track.id, undefined, { nbSamples: track.nb_samples })
      }
      mp4boxFile.start()
    }

    mp4boxFile.onSamples = (id: number, _user: unknown, samples: Sample[]) => {
      const track = collected.get(id)
      if (!track) return
      for (const sample of samples) {
        track.samples.push({
          data: sample.data!,
          duration: sample.duration,
          cts: sample.cts,
          dts: sample.dts,
          is_sync: sample.is_sync,
        })
      }
      // Hand ownership of the sample payloads over to `track.samples` above.
      // Without this mp4box.js holds its own reference to every sample for
      // the lifetime of the parse *and* can never drop the appended buffers
      // they came from, roughly doubling peak memory on files that are
      // already multi-gigabyte.
      const last = samples[samples.length - 1]
      if (last) mp4boxFile.releaseUsedSamples(id, last.number + 1)
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
export async function validateJoinCompatible(
  parts: File[],
  onProgress?: (fraction: number) => void,
): Promise<DemuxedFile[]> {
  if (parts.length < 2) throw new IncompatiblePartsError('need at least two video parts to join')

  // Sequentially, not Promise.all: every demuxed part is held in memory
  // until the authoring pass is done with it, so demuxing them all at once
  // would multiply an already-tight memory ceiling by the number of parts.
  // It also keeps reported progress monotonic instead of interleaved.
  const totalBytes = parts.reduce((sum, part) => sum + part.size, 0)
  let bytesDone = 0
  const demuxed: DemuxedFile[] = []
  for (const part of parts) {
    demuxed.push(await demuxFile(part, (fraction) => onProgress?.((bytesDone + fraction * part.size) / totalBytes)))
    bytesDone += part.size
  }

  const first = demuxed[0].tracks.map(signatureOf)
  for (let i = 1; i < demuxed.length; i++) {
    const sig = demuxed[i].tracks.map(signatureOf)
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
async function authorJoinedFile(
  demuxedParts: DemuxedFile[],
  onProgress?: (fraction: number) => void,
): Promise<Blob> {
  const out: ISOFile = createFile()
  // Explicitly, rather than letting the first `addTrack` call `init` with
  // that track's own options -- which would leave the movie inheriting the
  // video track's media timescale and a bogus mvhd volume.
  out.init({ timescale: MOVIE_TIMESCALE })

  // First-part track index -> output track id. Later parts' samples are
  // matched positionally, which validateJoinCompatible has already
  // confirmed lines up.
  const outputTrackIds: number[] = []
  // Per-track running offset, in that track's own timescale, to shift each
  // subsequent part's timestamps by. Accumulated from sample durations
  // (total decode duration) rather than from composition times, which run
  // out of order once there are B-frames.
  const timeOffsets: number[] = []

  for (const track of demuxedParts[0].tracks) {
    const { info, handler, description } = track
    const kind = trackKind(handler, info)
    const isVideo = kind === 'video'
    const outId = out.addTrack({
      type: PLACEHOLDER_FOURCC[kind],
      timescale: info.timescale,
      language: info.language,
      // `addTrack` defaults these to 320x320, which would put a nonsense
      // display size on a telemetry or audio track's tkhd.
      width: isVideo ? info.video?.width : 0,
      height: isVideo ? info.video?.height : 0,
      samplerate: info.audio?.sample_rate,
      channel_count: info.audio?.channel_count,
      samplesize: info.audio?.sample_size,
    })

    // Swap the placeholder sample entry and handler for the source's own.
    // Has to happen before any addSample call: addSample stamps each sample
    // with whatever stsd entry is in place at the time.
    const trak = out.getTrackById(outId)
    if (description) trak.mdia.minf.stbl.stsd.entries[0] = description
    if (handler) trak.mdia.hdlr.handler = handler

    outputTrackIds.push(outId)
    timeOffsets.push(0)
  }

  const totalSamples = demuxedParts.reduce(
    (sum, part) => sum + part.tracks.reduce((n, track) => n + track.samples.length, 0),
    0,
  )
  let samplesDone = 0

  for (const part of demuxedParts) {
    for (const [i, track] of part.tracks.entries()) {
      const outId = outputTrackIds[i]
      const offset = timeOffsets[i]
      let elapsed = 0
      for (const sample of track.samples) {
        out.addSample(outId, sample.data, {
          duration: sample.duration,
          cts: sample.cts + offset,
          dts: sample.dts + offset,
          is_sync: sample.is_sync,
        })
        elapsed += sample.duration
        if (++samplesDone % AUTHOR_YIELD_INTERVAL === 0) {
          onProgress?.(samplesDone / totalSamples)
          await yieldToEventLoop()
        }
      }
      timeOffsets[i] = offset + elapsed
    }
  }
  onProgress?.(1)

  // mp4box.js's authoring API never fills these in, leaving a file that
  // claims a duration of zero -- which sends probeVideoDuration down its
  // slowest fallback (walking every packet of a multi-gigabyte blob) and
  // makes the joined file look broken to anything that trusts the header.
  let movieDuration = 0
  for (const [i, outId] of outputTrackIds.entries()) {
    const trak = out.getTrackById(outId)
    const mediaDuration = timeOffsets[i]
    const timescale = trak.mdia.mdhd.timescale || 1
    const inMovieTime = Math.round((mediaDuration / timescale) * MOVIE_TIMESCALE)
    trak.mdia.mdhd.duration = mediaDuration
    trak.tkhd.duration = inMovieTime
    movieDuration = Math.max(movieDuration, inMovieTime)
  }
  out.moov.mvhd.duration = movieDuration

  // `save()` rather than `getBuffer()` was also triggering an actual browser
  // download of 'joined.mp4' as a side effect (it builds an <a download> and
  // clicks it), on top of returning the Blob this needs.
  return new Blob([out.getBuffer().buffer], { type: 'video/mp4' })
}

export async function joinVideos(parts: File[], onProgress?: (fraction: number) => void): Promise<Blob> {
  const demuxedParts = await validateJoinCompatible(parts, (fraction) =>
    onProgress?.(fraction * DEMUX_PROGRESS_SHARE),
  )
  return authorJoinedFile(demuxedParts, (fraction) =>
    onProgress?.(DEMUX_PROGRESS_SHARE + fraction * (1 - DEMUX_PROGRESS_SHARE)),
  )
}
