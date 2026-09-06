/** Main-thread facade for renderWorker.ts -- spawns the worker, ships it
 * the job, and adapts its postMessage protocol back to a normal
 * Promise + progress-callback shape for React components to consume. */
import type { OutputContainer } from '../env'
import type { AnnotatedRow } from '../laps/detection'
import type { RenderWorkerRequest, RenderWorkerResponse } from '../../workers/renderWorker'
import type { RenderStage } from './pipeline'
import type { RenderConfig } from './renderConfig'

export interface RunRenderOptions {
  videoFile: File
  trimStart: number
  trimEnd: number
  config: Omit<RenderConfig, 'widthPx' | 'heightPx'>
  annotatedRows: AnnotatedRow[]
  fileHandle?: FileSystemFileHandle
  outputContainer?: OutputContainer
  onProgress?: (fraction: number) => void
  onStage?: (stage: RenderStage) => void
}

export function runRender(options: RunRenderOptions): Promise<Blob | null> {
  const { onProgress, onStage, ...request } = options
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../../workers/renderWorker.ts', import.meta.url), { type: 'module' })

    worker.onmessage = (event: MessageEvent<RenderWorkerResponse>) => {
      const message = event.data
      if (message.type === 'progress') {
        onProgress?.(message.progress)
      } else if (message.type === 'stage') {
        onStage?.(message.stage)
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
    worker.postMessage(request satisfies RenderWorkerRequest)
  })
}
