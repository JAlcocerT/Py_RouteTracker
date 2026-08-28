import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { RenderWorkerRequest, RenderWorkerResponse } from './renderWorker'
import { DEFAULT_RENDER_CONFIG } from '../lib/render/renderConfig'

const renderVideo = vi.fn()
vi.mock('../lib/render/pipeline', () => ({ renderVideo: (...args: unknown[]) => renderVideo(...args) }))

/**
 * Stands in for a `FileSystemWritableFileStream`, tracking the writer lock the
 * real one inherits from `WritableStream`. mediabunny's StreamTarget takes a
 * writer on this stream and closes *that* writer to commit the file, which
 * leaves the stream locked from our side -- so `close()` here throws exactly
 * as the DOM does, which is the regression this file exists to catch.
 */
class FakeWritable {
  locked = false
  closed = false
  aborted = false

  close() {
    if (this.locked) return Promise.reject(new TypeError("Failed to execute 'close' on 'WritableStream': Cannot close a locked stream"))
    this.closed = true
    return Promise.resolve()
  }

  abort() {
    if (this.locked) return Promise.reject(new TypeError('Cannot abort a locked stream'))
    this.aborted = true
    return Promise.resolve()
  }

  /** What mediabunny's StreamTarget._start() does. */
  lock() {
    this.locked = true
  }
}

let writable: FakeWritable
const fileHandle = { createWritable: async () => writable } as unknown as FileSystemFileHandle

const row = { time: 0, lat: 0, lon: 0, speed: 0, lat_g: 0, lon_g: 0, lap: 0, last_lap_s: 0, lap_elapsed_s: 0 }
const request: RenderWorkerRequest = {
  videoFile: {} as File,
  trimStart: 0,
  trimEnd: 1,
  config: DEFAULT_RENDER_CONFIG,
  annotatedRows: [row],
  fileHandle,
}

/** Drives the worker's message handler and resolves with what it posts back. */
async function run(req: RenderWorkerRequest = request): Promise<RenderWorkerResponse[]> {
  const posted: RenderWorkerResponse[] = []
  vi.stubGlobal('postMessage', (message: RenderWorkerResponse) => posted.push(message))
  const self = globalThis as unknown as { onmessage: (e: MessageEvent<RenderWorkerRequest>) => Promise<void> }
  await self.onmessage({ data: req } as MessageEvent<RenderWorkerRequest>)
  return posted
}

beforeEach(async () => {
  vi.resetModules()
  renderVideo.mockReset()
  writable = new FakeWritable()
  vi.stubGlobal('postMessage', () => undefined)
  await import('./renderWorker')
})

describe('renderWorker', () => {
  it('does not close a stream mediabunny has already locked and closed', async () => {
    // The regression: every save-to-file render used to fail on its very last
    // step with "Cannot close a locked stream", after all the work was done.
    renderVideo.mockImplementation(async () => {
      writable.lock() // StreamTarget takes its writer
      return null // and mediabunny closes it, committing the file
    })

    const posted = await run()

    expect(posted).toEqual([{ type: 'done', blob: null }])
    expect(writable.closed).toBe(false)
  })

  it('reports the real error, not a cleanup failure, when a locked render fails', async () => {
    // Aborting a stream mediabunny still holds would throw on top of the
    // original error and bury the actual cause.
    renderVideo.mockImplementation(async () => {
      writable.lock()
      throw new Error('encoder blew up')
    })

    const posted = await run()

    expect(posted).toEqual([{ type: 'error', message: 'encoder blew up' }])
    expect(writable.aborted).toBe(false)
  })

  it('aborts the partial file when the render fails before mediabunny takes the stream', async () => {
    // Here the stream is still ours, so the half-written file the user picked
    // should be discarded rather than left holding a truncated video.
    renderVideo.mockRejectedValue(new Error('no telemetry in range'))

    const posted = await run()

    expect(posted).toEqual([{ type: 'error', message: 'no telemetry in range' }])
    expect(writable.aborted).toBe(true)
  })

  it('returns the blob when rendering in memory, with no stream involved', async () => {
    const blob = new Blob(['video'])
    renderVideo.mockResolvedValue(blob)

    const posted = await run({ ...request, fileHandle: undefined })

    expect(posted).toEqual([{ type: 'done', blob }])
  })

  it('forwards progress and stage updates as they happen', async () => {
    renderVideo.mockImplementation(async (_file: File, options: { onProgress?: (n: number) => void; onStage?: (s: string) => void }) => {
      options.onStage?.('software-decoding')
      options.onProgress?.(0.5)
      writable.lock()
      return null
    })

    const posted = await run()

    expect(posted).toEqual([{ type: 'stage', stage: 'software-decoding' }, { type: 'progress', progress: 0.5 }, { type: 'done', blob: null }])
  })
})
