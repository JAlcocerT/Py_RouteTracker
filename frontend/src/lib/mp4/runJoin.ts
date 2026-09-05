/** Main-thread facade for joinWorker.ts -- spawns the worker, ships it the
 * parts, and adapts its postMessage protocol back to a normal Promise +
 * progress-callback shape, mirroring render/runRender.ts. */
import type { JoinWorkerRequest, JoinWorkerResponse } from '../../workers/joinWorker'

export function runJoin(parts: File[], onProgress?: (fraction: number) => void): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../../workers/joinWorker.ts', import.meta.url), { type: 'module' })

    worker.onmessage = (event: MessageEvent<JoinWorkerResponse>) => {
      const message = event.data
      if (message.type === 'progress') {
        onProgress?.(message.progress)
      } else if (message.type === 'done') {
        worker.terminate()
        resolve(message.blob)
      } else {
        worker.terminate()
        reject(new Error(message.message))
      }
    }
    worker.onerror = (event) => {
      worker.terminate()
      reject(new Error(event.message))
    }
    worker.postMessage({ parts } satisfies JoinWorkerRequest)
  })
}
