/**
 * Lossless joining of split action-cam recordings. Ported from
 * backend/app/core/video_join.py, which used ffmpeg's concat demuxer to
 * stream-copy the parts together with no re-encode.
 *
 * This does the same at the packet level with mediabunny: read each part's
 * already-encoded packets out with an `EncodedPacketSink`, shift their
 * timestamps onto the joined timeline, and hand them straight to an
 * `EncodedVideoPacketSource`/`EncodedAudioPacketSource` on one `Output`.
 * No decoder or encoder is ever instantiated, so the video bytes that come
 * out are the ones that went in, and mediabunny writes a normal progressive
 * mp4 with real sample tables.
 *
 * On not preserving the `gpmd` track: mediabunny models video, audio and
 * subtitle tracks only (`ALL_TRACK_TYPES`), so a GoPro's telemetry track
 * cannot travel through this join -- the joined file carries picture and
 * sound, nothing else. That is deliberate rather than a regression. The
 * telemetry is read straight from the original parts instead, by
 * `extractGoProGpmfParts` (lib/telemetry/goproGpmf.ts), using the part start
 * times this function returns; the joined file's own telemetry track had no
 * other reader, and reading the camera's own files is the better-tested
 * path anyway.
 *
 * Size: pass an `outputStream` and the joined file is written straight out
 * as it's muxed, never held whole in memory. Without one the output goes to
 * a `BufferTarget`, which is a single `ArrayBuffer` -- and V8 caps those at
 * 4 GiB, so a join of any real length fails on the last byte with "ArrayBuffer
 * exceeded maximum size". joinWorker.ts always supplies a stream (onto an OPFS
 * scratch file) for exactly that reason; the bufferless path is kept only for
 * tests and for callers that know the result is small.
 *
 * The previous implementation authored the output with mp4box.js's
 * `addTrack`/`addSample` API, which *can* carry a `gpmd` track but is not
 * usable at this scale: `addSample` emits one `moof` + one `mdat` box per
 * sample (a 120-sample file came out as `ftyp moov moof x120 mdat x120`),
 * leaves the `moov`'s sample tables empty, and builds the whole thing in
 * memory on top of every demuxed sample of every part.
 */
import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  EncodedAudioPacketSource,
  EncodedPacket,
  EncodedPacketSink,
  EncodedVideoPacketSource,
  Input,
  type InputAudioTrack,
  type InputTrack,
  type InputVideoTrack,
  Mp4OutputFormat,
  Output,
  StreamTarget,
  type StreamTargetChunk,
} from 'mediabunny'

export class IncompatiblePartsError extends Error {}

export interface JoinVideosOptions {
  onProgress?: (fraction: number) => void
  /** Where to write the joined file. Given one, the result streams out and
   * `blob` comes back null -- the caller already has the destination. */
  outputStream?: WritableStream<StreamTargetChunk>
}

export interface JoinedVideo {
  /** The joined file, or null when it was streamed to an `outputStream`. */
  blob: Blob | null
  /** Where each input part begins in the joined timeline, in seconds. What
   * the telemetry read needs to line each part's samples up with the
   * joined footage. */
  partStarts: number[]
}

/** A part's tracks in the order they will be written, narrowed to the kinds
 * mediabunny can carry. */
interface PartTracks {
  video: InputVideoTrack[]
  audio: InputAudioTrack[]
}

function partition(tracks: InputTrack[]): PartTracks {
  return {
    video: tracks.filter((track): track is InputVideoTrack => track.isVideoTrack()),
    audio: tracks.filter((track): track is InputAudioTrack => track.isAudioTrack()),
  }
}

/** What has to match across parts for a lossless join to be possible: the
 * same tracks, in the same order, with the same codecs and geometry. */
function signatureOf({ video, audio }: PartTracks): string {
  return JSON.stringify([
    video.map((t) => [t.codec, t.displayWidth, t.displayHeight, t.rotation]),
    audio.map((t) => [t.codec, t.numberOfChannels, t.sampleRate]),
  ])
}

/** A track being read, holding the next packet due from it so the writer can
 * always pick the earliest one across all tracks. */
interface Cursor {
  packets: AsyncGenerator<EncodedPacket, void, unknown>
  pending: EncodedPacket | null
  add: (packet: EncodedPacket) => Promise<void>
}

async function openCursor(track: InputTrack, add: Cursor['add']): Promise<Cursor> {
  const packets = new EncodedPacketSink(track).packets()
  return { packets, pending: (await packets.next()).value ?? null, add }
}

export async function joinVideos(parts: File[], options: JoinVideosOptions = {}): Promise<JoinedVideo> {
  const { onProgress, outputStream } = options
  if (parts.length < 2) throw new IncompatiblePartsError('need at least two video parts to join')

  const inputs = parts.map((part) => new Input({ source: new BlobSource(part), formats: ALL_FORMATS }))
  const partTracks: PartTracks[] = []
  for (const [i, input] of inputs.entries()) {
    const tracks = partition(await input.getTracks())
    if (tracks.video.length === 0 && tracks.audio.length === 0) {
      throw new IncompatiblePartsError(`no video or audio tracks could be read from '${parts[i].name}'`)
    }
    partTracks.push(tracks)
    if (i > 0 && signatureOf(tracks) !== signatureOf(partTracks[0])) {
      throw new IncompatiblePartsError(
        `'${parts[i].name}' doesn't match the first part's format -- a lossless join needs every part ` +
          'to share the same tracks, codecs, and resolution. This usually means these files ' +
          "aren't chapters of the same recording, or one of them has been re-encoded.",
      )
    }
  }

  // Streaming out means the moov can only go last: its size isn't known until
  // the final packet is written, and 'in-memory' faststart would buffer the
  // whole file to move it to the front -- the very thing the stream avoids.
  // Moov-last costs nothing here, because every reader of this file (probe.ts,
  // the render pipeline) reads it as a local File and can seek to the end.
  // Buffered, faststart is free, since BufferTarget holds it all regardless.
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: outputStream ? false : 'in-memory' }),
    // chunked batches the writes into 16 MiB blocks rather than issuing one
    // per packet, which matters when the destination is a real file.
    target: outputStream ? new StreamTarget(outputStream, { chunked: true }) : new BufferTarget(),
  })

  // Track layout is taken from the first part, which every other part has
  // just been checked against, and the decoder configs are declared up front
  // so the muxer has full track information before the first packet.
  const videoSources: EncodedVideoPacketSource[] = []
  const audioSources: EncodedAudioPacketSource[] = []
  for (const track of partTracks[0].video) {
    const source = new EncodedVideoPacketSource(track.codec!)
    output.addVideoTrack(source, {
      rotation: track.rotation,
      decoderConfig: (await track.getDecoderConfig()) ?? undefined,
    })
    videoSources.push(source)
  }
  for (const track of partTracks[0].audio) {
    const source = new EncodedAudioPacketSource(track.codec!)
    output.addAudioTrack(source, { decoderConfig: (await track.getDecoderConfig()) ?? undefined })
    audioSources.push(source)
  }
  await output.start()

  const totalBytes = parts.reduce((sum, part) => sum + part.size, 0)
  let bytesDone = 0
  let offset = 0
  let reported = 0
  const partStarts: number[] = []

  // Progress can only ever go forwards. Within a part it is driven by
  // timestamps, which jitter a little when the tracks interleave unevenly.
  const report = (fraction: number) => {
    if (fraction <= reported) return
    reported = Math.min(1, fraction)
    onProgress?.(reported)
  }

  for (const [i, tracks] of partTracks.entries()) {
    partStarts.push(offset)
    // Only used to animate progress within this part; a part whose container
    // doesn't state a duration just doesn't move the bar until it finishes.
    const partDuration = await inputs[i].getDurationFromMetadata()
    let partEnd = 0

    // One cursor per track, advanced lowest-timestamp-first so the packets
    // reach the muxer roughly interleaved. Writing one whole track and then
    // the next would work, but it makes the muxer hold everything it can't
    // commit yet, and it makes progress lurch backwards at each track change.
    const cursors = await Promise.all([
      ...tracks.video.map((track, k) => openCursor(track, (p) => videoSources[k].add(p))),
      ...tracks.audio.map((track, k) => openCursor(track, (p) => audioSources[k].add(p))),
    ])

    for (;;) {
      let next: Cursor | undefined
      for (const cursor of cursors) {
        if (cursor.pending && (!next || cursor.pending.timestamp < next.pending!.timestamp)) next = cursor
      }
      if (!next) break

      const packet = next.pending!
      // Packets come out in decode order carrying presentation timestamps,
      // which is exactly what `add` wants, so shifting the timestamp is the
      // whole of the concatenation -- the payload is passed through untouched.
      await next.add(
        new EncodedPacket(packet.data, packet.type, packet.timestamp + offset, packet.duration, packet.sequenceNumber),
      )
      partEnd = Math.max(partEnd, packet.timestamp + packet.duration)
      next.pending = (await next.packets.next()).value ?? null

      if (partDuration) {
        const within = Math.min(1, (packet.timestamp + packet.duration) / partDuration)
        report((bytesDone + within * parts[i].size) / totalBytes)
      }
    }

    bytesDone += parts[i].size
    report(bytesDone / totalBytes)
    // Advance every track by the same amount -- the part's own length -- so
    // audio and video stay in sync across the seam even when one of them
    // runs marginally shorter than the other.
    offset += partEnd
  }

  await output.finalize()
  report(1)
  if (outputStream) return { blob: null, partStarts }
  return { blob: new Blob([(output.target as BufferTarget).buffer!], { type: 'video/mp4' }), partStarts }
}
