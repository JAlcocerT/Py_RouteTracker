/** Fixtures are authored with mp4box.js itself. The one thing it cannot
 * author is a `gpmd` track -- GoPro's telemetry fourcc isn't in its sample
 * entry registry at all, which is exactly why this module has to copy the
 * source's sample entry verbatim rather than let mp4box.js rebuild one. So
 * the metadata track is authored as `mp4s` (same box shape, registered) and
 * its fourcc patched to `gpmd` in the raw bytes, giving a parseable
 * stand-in for the real thing. */
import { describe, expect, it, vi } from 'vitest'
// Node's File, not jsdom's: `demuxFile` reads the part via `File.stream()`,
// which every browser implements and jsdom does not.
import { File as NodeFile } from 'node:buffer'
import { createFile, type Movie } from 'mp4box'
import { IncompatiblePartsError, joinVideos } from './join'

const SAMPLES_PER_TRACK = 30
const SAMPLE_DURATION = 100
// Big enough that a part spans several `File.stream()` chunks, so the demux
// (and the progress it reports, and the sample-release it does between
// chunks) is exercised incrementally rather than in a single append.
const SAMPLE_BYTES = 4096

/** A GoPro-shaped part: HEVC video + AAC audio + a `gpmd` telemetry track. */
function buildPart(name: string, { videoFourcc = 'hvc1' }: { videoFourcc?: 'hvc1' | 'avc1' } = {}): File {
  const file = createFile()
  const tracks = [
    { type: videoFourcc, hdlr: 'vide', timescale: 1000, width: 64, height: 64 },
    { type: 'mp4a', hdlr: 'soun', timescale: 1000, samplerate: 48000, channel_count: 2, samplesize: 16 },
    { type: 'mp4s', hdlr: 'meta', timescale: 1000 },
  ] as const

  for (const options of tracks) {
    const id = file.addTrack(options)
    for (let i = 0; i < SAMPLES_PER_TRACK; i++) {
      file.addSample(id, new Uint8Array(SAMPLE_BYTES).fill(i), {
        duration: SAMPLE_DURATION,
        cts: i * SAMPLE_DURATION,
        dts: i * SAMPLE_DURATION,
        is_sync: i === 0,
      })
    }
  }

  const bytes = new Uint8Array(file.getBuffer().buffer)
  patchFourcc(bytes, 'mp4s', 'gpmd')
  return new NodeFile([bytes], name, { type: 'video/mp4' }) as unknown as File
}

/** Rewrites a fourcc wherever it appears in the raw bytes. Safe only because
 * the fourcc it is called with (`mp4s`) occurs exactly once in the authored
 * fixture -- as that track's sample entry type. */
function patchFourcc(bytes: Uint8Array, from: string, to: string): void {
  const [needle, replacement] = [from, to].map((s) => [...s].map((c) => c.charCodeAt(0)))
  for (let i = 0; i <= bytes.length - 4; i++) {
    if (needle.every((byte, k) => bytes[i + k] === byte)) bytes.set(replacement, i)
  }
}

interface ParsedTrack {
  codec: string
  handler: string
  nbSamples: number
  cts: number[]
  /** Each sample's payload reduced to (size, first byte) -- buildPart fills
   * sample *i* with the byte *i*, so this is enough to prove the bytes came
   * through intact and in order. */
  payloads: Array<[number, number]>
}

async function parse(blob: Blob): Promise<{ duration: number; timescale: number; tracks: ParsedTrack[] }> {
  const file = createFile()
  let movie: Movie | undefined
  const cts = new Map<number, number[]>()
  const payloads = new Map<number, Array<[number, number]>>()
  file.onReady = (info) => {
    movie = info
    for (const track of info.tracks) {
      cts.set(track.id, [])
      payloads.set(track.id, [])
      file.setExtractionOptions(track.id, undefined, { nbSamples: track.nb_samples })
    }
    file.start()
  }
  file.onSamples = (id, _user, samples) => {
    cts.get(id)?.push(...samples.map((s) => s.cts))
    payloads.get(id)?.push(...samples.map((s): [number, number] => [s.size, s.data![0]]))
  }

  const buffer = await blob.arrayBuffer()
  Object.assign(buffer, { fileStart: 0 })
  file.appendBuffer(buffer as ArrayBuffer & { fileStart: number })
  file.flush()

  if (!movie) throw new Error('joined output did not parse')
  return {
    duration: movie.duration,
    timescale: movie.timescale,
    tracks: movie.tracks.map((track) => ({
      codec: track.codec,
      handler: file.getTrackById(track.id).mdia.hdlr.handler,
      nbSamples: track.nb_samples,
      cts: cts.get(track.id) ?? [],
      payloads: payloads.get(track.id) ?? [],
    })),
  }
}

describe('joinVideos', () => {
  it('concatenates every track and leaves timestamps contiguous', async () => {
    const joined = await parse(await joinVideos([buildPart('GH010001.MP4'), buildPart('GH020001.MP4')]))

    expect(joined.tracks).toHaveLength(3)
    for (const track of joined.tracks) {
      expect(track.nbSamples).toBe(SAMPLES_PER_TRACK * 2)
      // The second part picks up exactly where the first left off -- no gap,
      // no overlap, no reset to zero.
      expect(track.cts).toEqual(
        Array.from({ length: SAMPLES_PER_TRACK * 2 }, (_, i) => i * SAMPLE_DURATION),
      )
      // ...and every sample's bytes survived the copy, in order. This is
      // what a "lossless" join has to mean, and it is also what would break
      // if the demuxer released mp4box.js's sample store too eagerly.
      expect(track.payloads).toEqual(
        Array.from({ length: SAMPLES_PER_TRACK * 2 }, (_, i) => [SAMPLE_BYTES, i % SAMPLES_PER_TRACK]),
      )
    }
  })

  it("preserves each track's own codec and handler", async () => {
    // The regression this guards: mp4box.js's `addTrack` builds its sample
    // entry from `options.type`, defaulting to `avc1`. Omitting the option
    // and passing the source entry as `description` nested it inside a fresh
    // `avc1` entry instead of using it, so every track in the joined file --
    // HEVC video, AAC audio, and the gpmd telemetry track alike -- came out
    // as an `avc1` video track.
    const joined = await parse(await joinVideos([buildPart('GH010001.MP4'), buildPart('GH020001.MP4')]))

    expect(joined.tracks.map((t) => t.codec)).toEqual(['hvc1', 'mp4a', 'gpmd'])
    expect(joined.tracks.map((t) => t.handler)).toEqual(['vide', 'soun', 'meta'])
  })

  it('states a duration in the output header', async () => {
    // mp4box.js's authoring API leaves mvhd/tkhd/mdhd duration at zero,
    // which makes the joined file look zero-length to anything reading the
    // header and pushes probeVideoDuration onto its walk-every-packet path.
    const joined = await parse(await joinVideos([buildPart('GH010001.MP4'), buildPart('GH020001.MP4')]))

    const expectedSeconds = (SAMPLES_PER_TRACK * 2 * SAMPLE_DURATION) / 1000
    expect(joined.duration / joined.timescale).toBeCloseTo(expectedSeconds, 3)
  })

  it('reports progress as it goes, not only on completion', async () => {
    const seen: number[] = []
    await joinVideos([buildPart('GH010001.MP4'), buildPart('GH020001.MP4')], (f) => seen.push(f))

    // The bug users saw: `joinVideos` accepted an onProgress callback but
    // never handed it to the demuxer, so the bar sat at 0% for the entire
    // join and then jumped straight to 100%.
    expect(seen.length).toBeGreaterThan(1)
    expect(seen[0]).toBeGreaterThan(0)
    expect(seen[0]).toBeLessThan(1)
    expect(seen).toEqual([...seen].sort((a, b) => a - b))
    expect(seen.at(-1)).toBe(1)
  })

  it('does not trigger a file download as a side effect', async () => {
    // mp4box.js's `ISOFile.save()` returns the Blob *and* clicks an
    // <a download> for it, which popped an unasked-for 'joined.mp4' save.
    const createObjectURL = vi.spyOn(URL, 'createObjectURL')
    await joinVideos([buildPart('GH010001.MP4'), buildPart('GH020001.MP4')])
    expect(createObjectURL).not.toHaveBeenCalled()
    createObjectURL.mockRestore()
  })

  it('rejects parts whose track layouts differ', async () => {
    const parts = [buildPart('GH010001.MP4'), buildPart('other.MP4', { videoFourcc: 'avc1' })]
    await expect(joinVideos(parts)).rejects.toThrow(IncompatiblePartsError)
  })

  it('rejects a single part', async () => {
    await expect(joinVideos([buildPart('GH010001.MP4')])).rejects.toThrow(IncompatiblePartsError)
  })
})
