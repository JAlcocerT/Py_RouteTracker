/**
 * Fixtures are progressive mp4s built with mediabunny: a single `mdat` with
 * the `moov` *after* it, which is how a camera writes a recording (it can't
 * know the sample tables until it stops). Packet payloads are filled with a
 * recognisable byte per frame so the join can be checked for actually being
 * lossless rather than merely producing a file.
 */
import { describe, expect, it, vi } from 'vitest'
import {
  ALL_FORMATS, BlobSource, BufferTarget, EncodedAudioPacketSource, EncodedPacket, EncodedPacketSink,
  EncodedVideoPacketSource, Input, Mp4OutputFormat, Output, type StreamTargetChunk,
} from 'mediabunny'
import { IncompatiblePartsError, type JoinVideosOptions, joinVideos } from './join'

const FRAMES = 120
const FPS = 30
const VIDEO_BYTES = 4096
const AUDIO_BYTES = 256
/** Minimal but valid AVCDecoderConfigurationRecord. */
const AVCC = new Uint8Array([1, 66, 0, 10, 255, 225, 0, 3, 103, 66, 0, 1, 0, 3, 104, 206, 1, 0])
/** AudioSpecificConfig: AAC-LC, 48 kHz, stereo. */
const ASC = new Uint8Array([0x11, 0x90])

interface PartOptions {
  frames?: number
  withAudio?: boolean
  width?: number
  /** Byte written into every packet of this part, so payloads are traceable
   * back to the part they came from. */
  fill?: number
}

async function buildPart(name: string, options: PartOptions = {}): Promise<File> {
  const { frames = FRAMES, withAudio = true, width = 1920, fill } = options
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: false }), target: new BufferTarget() })
  const video = new EncodedVideoPacketSource('avc')
  const audio = new EncodedAudioPacketSource('aac')
  output.addVideoTrack(video, { rotation: 180 })
  if (withAudio) output.addAudioTrack(audio)
  await output.start()

  for (let i = 0; i < frames; i++) {
    const byte = fill ?? i % 251
    await video.add(
      new EncodedPacket(new Uint8Array(VIDEO_BYTES).fill(byte), i % 30 === 0 ? 'key' : 'delta', i / FPS, 1 / FPS),
      i === 0
        ? { decoderConfig: { codec: 'avc1.42000a', codedWidth: width, codedHeight: 1080, description: AVCC } }
        : undefined,
    )
    if (withAudio) {
      await audio.add(
        new EncodedPacket(new Uint8Array(AUDIO_BYTES).fill(byte), 'key', i / FPS, 1 / FPS),
        i === 0
          ? { decoderConfig: { codec: 'mp4a.40.2', numberOfChannels: 2, sampleRate: 48000, description: ASC } }
          : undefined,
      )
    }
  }
  await output.finalize()
  return new File([new Uint8Array(output.target.buffer!)], name, { type: 'video/mp4' })
}

interface ReadTrack {
  type: string
  codec: string | null
  rotation: number | null
  timestamps: number[]
  /** (byteLength, first byte) per packet -- enough to prove the payloads came
   * through untouched and in order. */
  payloads: Array<[number, number]>
}

async function read(blob: Blob): Promise<{ duration: number | null; tracks: ReadTrack[] }> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS })
  const tracks: ReadTrack[] = []
  for (const track of await input.getTracks()) {
    const timestamps: number[] = []
    const payloads: Array<[number, number]> = []
    for await (const packet of new EncodedPacketSink(track).packets()) {
      timestamps.push(packet.timestamp)
      payloads.push([packet.data.length, packet.data[0]])
    }
    tracks.push({
      type: track.type,
      codec: track.codec,
      rotation: track.isVideoTrack() ? track.rotation : null,
      timestamps,
      payloads,
    })
  }
  return { duration: await input.getDurationFromMetadata(), tracks }
}

/** joinVideos on its buffered path, with the now-nullable blob narrowed. */
async function joinToBlob(parts: File[], options: JoinVideosOptions = {}) {
  const { blob, partStarts } = await joinVideos(parts, options)
  if (!blob) throw new Error('expected a buffered join to return a blob')
  return { blob, partStarts }
}

/** Stands in for a FileSystemWritableFileStream: honours the positioned
 * writes mediabunny's StreamTarget issues and reassembles the file. */
function collectingStream() {
  const writes: { data: Uint8Array; position: number }[] = []
  let size = 0
  const stream = new WritableStream<StreamTargetChunk>({
    write(chunk) {
      writes.push({ data: new Uint8Array(chunk.data), position: chunk.position })
      size = Math.max(size, chunk.position + chunk.data.byteLength)
    },
  })
  const assembled = () => {
    const out = new Uint8Array(size)
    for (const write of writes) out.set(write.data, write.position)
    return out
  }
  return { stream, assembled, writeCount: () => writes.length }
}

describe('joinVideos', () => {
  it('passes every packet through byte for byte, in order', async () => {
    const parts = [await buildPart('GH010433.MP4', { fill: 7 }), await buildPart('GH020433.MP4', { fill: 9 })]
    const { blob } = await joinToBlob(parts)
    const joined = await read(blob)

    expect(joined.tracks.map((t) => `${t.type}/${t.codec}`)).toEqual(['video/avc', 'audio/aac'])
    for (const track of joined.tracks) {
      const size = track.type === 'video' ? VIDEO_BYTES : AUDIO_BYTES
      expect(track.payloads).toEqual([
        ...Array.from({ length: FRAMES }, (): [number, number] => [size, 7]),
        ...Array.from({ length: FRAMES }, (): [number, number] => [size, 9]),
      ])
    }
  }, 30000)

  it('lays the second part end to end after the first', async () => {
    const parts = [await buildPart('GH010433.MP4'), await buildPart('GH020433.MP4')]
    const { blob, partStarts } = await joinToBlob(parts)
    const joined = await read(blob)

    expect(partStarts).toEqual([0, FRAMES / FPS])
    expect(joined.duration).toBeCloseTo((FRAMES * 2) / FPS, 3)
    for (const track of joined.tracks) {
      expect(track.timestamps).toEqual(
        Array.from({ length: FRAMES * 2 }, (_, i) => expect.closeTo(i / FPS, 4) as unknown as number),
      )
    }
  }, 30000)

  it('keeps audio and video aligned when parts are of unequal length', async () => {
    const parts = [
      await buildPart('GH010433.MP4', { frames: 90 }),
      await buildPart('GH020433.MP4', { frames: 40 }),
      await buildPart('GH030433.MP4', { frames: 65 }),
    ]
    const { blob, partStarts } = await joinToBlob(parts)
    const joined = await read(blob)

    expect(partStarts[0]).toBe(0)
    expect(partStarts[1]).toBeCloseTo(90 / FPS, 6)
    expect(partStarts[2]).toBeCloseTo(130 / FPS, 6)
    // Both tracks carry every packet and start each part at the same instant,
    // which is what stops the sound drifting away from the picture at a seam.
    for (const track of joined.tracks) expect(track.timestamps).toHaveLength(195)
    expect(joined.tracks[0].timestamps).toEqual(joined.tracks[1].timestamps)
  }, 30000)

  it('preserves track rotation', async () => {
    const { blob } = await joinToBlob([await buildPart('a.MP4'), await buildPart('b.MP4')])
    expect((await read(blob)).tracks[0].rotation).toBe(180)
  }, 30000)

  it('reports progress as it goes, not only on completion', async () => {
    const seen: number[] = []
    await joinVideos([await buildPart('a.MP4'), await buildPart('b.MP4')], { onProgress: (f) => seen.push(f) })

    expect(seen.length).toBeGreaterThan(1)
    expect(seen[0]).toBeGreaterThan(0)
    expect(seen[0]).toBeLessThan(1)
    expect(seen).toEqual([...seen].sort((a, b) => a - b))
    expect(seen.at(-1)).toBe(1)
  }, 30000)

  it('does not trigger a file download as a side effect', async () => {
    const createObjectURL = vi.spyOn(URL, 'createObjectURL')
    await joinVideos([await buildPart('a.MP4'), await buildPart('b.MP4')])
    expect(createObjectURL).not.toHaveBeenCalled()
    createObjectURL.mockRestore()
  }, 30000)

  it('streams the joined file out rather than buffering it when given an outputStream', async () => {
    const parts = [await buildPart('GH010433.MP4', { fill: 7 }), await buildPart('GH020433.MP4', { fill: 9 })]
    const sink = collectingStream()

    const { blob, partStarts } = await joinVideos(parts, { outputStream: sink.stream })

    // Nothing came back in memory: this is the whole point -- a real joined
    // recording is bigger than the 4 GiB an ArrayBuffer can hold.
    expect(blob).toBeNull()
    expect(partStarts).toEqual([0, FRAMES / FPS])

    // What landed in the stream is a complete, readable mp4 carrying exactly
    // the payloads the buffered path produces. The destination changed; the
    // bytes did not.
    const streamed = await read(new Blob([sink.assembled()], { type: 'video/mp4' }))
    expect(streamed.duration).toBeCloseTo((FRAMES * 2) / FPS, 3)
    for (const track of streamed.tracks) {
      const size = track.type === 'video' ? VIDEO_BYTES : AUDIO_BYTES
      expect(track.payloads).toEqual([
        ...Array.from({ length: FRAMES }, (): [number, number] => [size, 7]),
        ...Array.from({ length: FRAMES }, (): [number, number] => [size, 9]),
      ])
    }
  }, 30000)

  it('batches stream writes rather than issuing one per packet', async () => {
    const parts = [await buildPart('GH010433.MP4'), await buildPart('GH020433.MP4')]
    const sink = collectingStream()

    await joinVideos(parts, { outputStream: sink.stream })

    // 480 packets go in; chunking means far fewer writes come out, which is
    // what keeps a join from thrashing the filesystem.
    expect(sink.writeCount()).toBeLessThan(FRAMES)
  }, 30000)

  it('rejects parts whose track layouts differ', async () => {
    const parts = [await buildPart('GH010433.MP4'), await buildPart('other.MP4', { withAudio: false })]
    await expect(joinVideos(parts)).rejects.toThrow(IncompatiblePartsError)
  }, 30000)

  it('rejects parts whose resolutions differ', async () => {
    const parts = [await buildPart('GH010433.MP4'), await buildPart('other.MP4', { width: 1280 })]
    await expect(joinVideos(parts)).rejects.toThrow(IncompatiblePartsError)
  }, 30000)

  it('rejects a single part', async () => {
    await expect(joinVideos([await buildPart('GH010433.MP4')])).rejects.toThrow(IncompatiblePartsError)
  }, 30000)

  it('rejects a part with no readable media', async () => {
    const full = await buildPart('GH010433.MP4')
    const truncated = new File([await full.slice(0, 2048).arrayBuffer()], 'truncated.MP4', { type: 'video/mp4' })
    await expect(joinVideos([truncated, full])).rejects.toThrow(/truncated.MP4/)
  }, 30000)
})
