/** Hosts the render pipeline (WebCodecs decode + Canvas2D HUD draw + encode
 * + mux) off the main thread, so a multi-minute render doesn't freeze the
 * UI -- the same reason the server version ran this in a background
 * thread/process (see app.render.local_worker's own comment on exactly
 * this). Runs standalone: it can't hold a `FileSystemWritableFileStream`
 * across the postMessage boundary (not structured-cloneable), so a
 * `FileSystemFileHandle` is passed instead and turned into a writable here. */
import type { AnnotatedRow } from '../lib/laps/detection'
import { renderVideo } from '../lib/render/pipeline'
import type { RenderConfig } from '../lib/render/renderConfig'

export interface RenderWorkerRequest {
  videoFile: File
  trimStart: number
  trimEnd: number
  config: Omit<RenderConfig, 'widthPx' | 'heightPx'>
  annotatedRows: AnnotatedRow[]
  fileHandle?: FileSystemFileHandle
}

export type RenderWorkerResponse =
  | { type: 'progress'; progress: number }
  | { type: 'done'; blob: Blob | null }
  | { type: 'error'; message: string }

self.onmessage = async (event: MessageEvent<RenderWorkerRequest>) => {
  const { videoFile, trimStart, trimEnd, config, annotatedRows, fileHandle } = event.data
  try {
    const outputStream = fileHandle ? await fileHandle.createWritable() : undefined
    const blob = await renderVideo(videoFile, {
      trimStart,
      trimEnd,
      config,
      annotatedRows,
      outputStream,
      onProgress: (progress) => {
        const message: RenderWorkerResponse = { type: 'progress', progress }
        self.postMessage(message)
      },
    })
    if (outputStream) await outputStream.close()
    const message: RenderWorkerResponse = { type: 'done', blob }
    self.postMessage(message)
  } catch (error) {
    const message: RenderWorkerResponse = { type: 'error', message: error instanceof Error ? error.message : String(error) }
    self.postMessage(message)
  }
}
