/**
 * The worker owns where a join's bytes land, so that's most of what's worth
 * testing here: that it streams to an OPFS scratch file rather than
 * buffering (a real joined recording exceeds the 4 GiB ArrayBuffer cap), and
 * that it still works where OPFS can't be written to.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { JoinWorkerRequest, JoinWorkerResponse } from './joinWorker'

const joinVideos = vi.fn()
vi.mock('../lib/mp4/join', () => ({ joinVideos: (...args: unknown[]) => joinVideos(...args) }))

const parts = [
  { name: 'GH010001.MP4', size: 2_000_000_000 } as File,
  { name: 'GH020001.MP4', size: 2_000_000_000 } as File,
]

let writable: { locked: boolean; abort: ReturnType<typeof vi.fn>; close: ReturnType<typeof vi.fn> }
let scratchFile: File
let fileHandle: { createWritable: ReturnType<typeof vi.fn>; getFile: ReturnType<typeof vi.fn> }
let directory: { getFileHandle: ReturnType<typeof vi.fn> }
let root: { removeEntry: ReturnType<typeof vi.fn>; getDirectoryHandle: ReturnType<typeof vi.fn> }
let estimate: ReturnType<typeof vi.fn>

/** An OPFS that works. */
function stubOpfs() {
  writable = { locked: false, abort: vi.fn(async () => undefined), close: vi.fn(async () => undefined) }
  scratchFile = new File(['joined'], 'GH010001.MP4', { type: 'video/mp4' })
  fileHandle = { createWritable: vi.fn(async () => writable), getFile: vi.fn(async () => scratchFile) }
  directory = { getFileHandle: vi.fn(async () => fileHandle) }
  root = { removeEntry: vi.fn(async () => undefined), getDirectoryHandle: vi.fn(async () => directory) }
  estimate = vi.fn(async () => ({ quota: 40_000_000_000, usage: 1_000_000_000 }))
  vi.stubGlobal('navigator', { storage: { getDirectory: async () => root, estimate } })
}

/** Drives the worker's message handler and resolves with what it posts back. */
async function run(request: JoinWorkerRequest = { parts }): Promise<JoinWorkerResponse[]> {
  const posted: JoinWorkerResponse[] = []
  vi.stubGlobal('postMessage', (message: JoinWorkerResponse) => posted.push(message))
  const self = globalThis as unknown as { onmessage: (e: MessageEvent<JoinWorkerRequest>) => Promise<void> }
  await self.onmessage({ data: request } as MessageEvent<JoinWorkerRequest>)
  return posted
}

beforeEach(async () => {
  vi.resetModules()
  joinVideos.mockReset()
  joinVideos.mockResolvedValue({ blob: null, partStarts: [0, 12.5] })
  stubOpfs()
  vi.stubGlobal('postMessage', () => undefined)
  await import('./joinWorker')
})

describe('joinWorker', () => {
  it('streams the join into an OPFS scratch file and hands back that file', async () => {
    const posted = await run()

    expect(joinVideos).toHaveBeenCalledWith(parts, {
      outputStream: writable,
      onProgress: expect.any(Function),
    })
    // Named after the first part, because that name is what the rest of the
    // app shows the user for the joined recording.
    expect(directory.getFileHandle).toHaveBeenCalledWith('GH010001.MP4', { create: true })
    expect(posted).toEqual([
      // partStarts has to survive the postMessage hop: the telemetry read
      // needs it to place each part's samples on the joined timeline.
      { type: 'done', joined: { file: scratchFile, partStarts: [0, 12.5] } },
    ])
  })

  it('clears the previous join before starting a new one', async () => {
    await run()

    // These files are whole-recording sized, so leaving them behind would
    // quietly eat the origin's storage quota.
    expect(root.removeEntry).toHaveBeenCalledWith('joins', { recursive: true })
    expect(root.removeEntry.mock.invocationCallOrder[0]).toBeLessThan(
      directory.getFileHandle.mock.invocationCallOrder[0],
    )
  })

  it('does not close the writable itself', async () => {
    await run()

    // mediabunny's StreamTarget holds a writer on it and closes that on
    // finalize; for a FileSystemWritableFileStream that close is what commits
    // the bytes. Closing again here throws "Cannot close a locked stream".
    expect(writable.close).not.toHaveBeenCalled()
    expect(writable.abort).not.toHaveBeenCalled()
  })

  it('falls back to buffering where OPFS cannot be written', async () => {
    vi.stubGlobal('navigator', {})
    joinVideos.mockResolvedValue({ blob: new Blob(['joined']), partStarts: [0, 8] })

    const posted = await run()

    expect(joinVideos).toHaveBeenCalledWith(parts, {
      outputStream: undefined,
      onProgress: expect.any(Function),
    })
    const done = posted[0] as Extract<JoinWorkerResponse, { type: 'done' }>
    expect(done.joined.file.name).toBe('GH010001.MP4')
    expect(done.joined.file.type).toBe('video/mp4')
    expect(done.joined.partStarts).toEqual([0, 8])
  })

  it('forwards progress as it arrives, then the joined video', async () => {
    joinVideos.mockImplementation(async (_parts: File[], { onProgress }: { onProgress: (f: number) => void }) => {
      onProgress(0.5)
      onProgress(1)
      return { blob: null, partStarts: [0, 12.5] }
    })

    const posted = await run()

    expect(posted).toEqual([
      { type: 'progress', progress: 0.5 },
      { type: 'progress', progress: 1 },
      { type: 'done', joined: { file: scratchFile, partStarts: [0, 12.5] } },
    ])
  })

  it('refuses up front when the origin has no room for the joined file', async () => {
    // 4 GB of parts, 1.5 GB free. Better to say so now than to find out an
    // hour into a join.
    estimate.mockResolvedValue({ quota: 10_000_000_000, usage: 8_500_000_000 })

    const posted = await run()

    expect(joinVideos).not.toHaveBeenCalled()
    const error = posted[0] as Extract<JoinWorkerResponse, { type: 'error' }>
    expect(error.type).toBe('error')
    expect(error.message).toMatch(/4\.0 GB of scratch space and only 1\.5 GB is free/)
  })

  it('joins anyway when the browser will not estimate storage', async () => {
    estimate.mockResolvedValue({})

    await run()

    // Refusing on a number we do not have would block joins that would work.
    expect(joinVideos).toHaveBeenCalled()
  })

  it('explains a quota blowout that happens mid-join', async () => {
    joinVideos.mockRejectedValue(new DOMException('boom', 'QuotaExceededError'))

    const posted = await run()

    // "QuotaExceededError" alone tells the user nothing about what to do.
    expect(posted).toEqual([
      {
        type: 'error',
        message:
          "Ran out of browser storage part way through the join. Clear some disk space, or this site's " +
          'stored data, and try again.',
      },
    ])
  })

  it('reports a failed join as an error message rather than an unhandled rejection', async () => {
    joinVideos.mockRejectedValue(new Error("'GH020001.MP4' doesn't match the first part's format"))

    const posted = await run()

    expect(posted).toEqual([
      { type: 'error', message: "'GH020001.MP4' doesn't match the first part's format" },
    ])
  })

  it('discards the half-written scratch file when the join fails', async () => {
    joinVideos.mockRejectedValue(new Error('boom'))

    await run()

    // Better an absent file than one holding a truncated video.
    expect(writable.abort).toHaveBeenCalled()
  })

  it('leaves the stream alone on failure once mediabunny holds the writer', async () => {
    joinVideos.mockImplementation(async () => {
      writable.locked = true
      throw new Error('boom')
    })

    const posted = await run()

    // Aborting a locked stream throws, which would bury the real error.
    expect(writable.abort).not.toHaveBeenCalled()
    expect(posted).toEqual([{ type: 'error', message: 'boom' }])
  })
})
