import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { JoinWorkerRequest, JoinWorkerResponse } from './joinWorker'

const joinVideos = vi.fn()
vi.mock('../lib/mp4/join', () => ({ joinVideos: (...args: unknown[]) => joinVideos(...args) }))

const parts = [{ name: 'GH010001.MP4' } as File, { name: 'GH020001.MP4' } as File]

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
  vi.stubGlobal('postMessage', () => undefined)
  await import('./joinWorker')
})

describe('joinWorker', () => {
  it('forwards progress as it arrives, then the joined blob', async () => {
    const blob = new Blob(['joined'])
    joinVideos.mockImplementation(async (_parts: File[], onProgress: (f: number) => void) => {
      onProgress(0.5)
      onProgress(1)
      return blob
    })

    const posted = await run()

    expect(joinVideos).toHaveBeenCalledWith(parts, expect.any(Function))
    expect(posted).toEqual([
      { type: 'progress', progress: 0.5 },
      { type: 'progress', progress: 1 },
      { type: 'done', blob },
    ])
  })

  it('reports a failed join as an error message rather than an unhandled rejection', async () => {
    joinVideos.mockRejectedValue(new Error("'GH020001.MP4' doesn't match the first part's format"))

    const posted = await run()

    expect(posted).toEqual([
      { type: 'error', message: "'GH020001.MP4' doesn't match the first part's format" },
    ])
  })
})
