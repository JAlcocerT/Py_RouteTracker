/** Hosts the render pipeline (WebCodecs decode + Canvas2D HUD draw + encode
 * + mux) off the main thread, so a multi-minute render doesn't freeze the
 * UI -- the same reason the server version ran this in a background
 * thread/process (see app.render.local_worker's own comment on exactly
 * this). Runs standalone: it can't hold a `FileSystemWritableFileStream`
 * across the postMessage boundary (not structured-cloneable), so a
 * `FileSystemFileHandle` is passed instead and turned into a writable here. */
import type { OutputContainer } from '../lib/env'
import type { AnnotatedRow } from '../lib/laps/detection'
import { renderVideo, type RenderStage } from '../lib/render/pipeline'
import type { RenderConfig } from '../lib/render/renderConfig'

export interface RenderWorkerRequest {
  videoFile: File
  trimStart: number
  trimEnd: number
  config: Omit<RenderConfig, 'widthPx' | 'heightPx'>
  annotatedRows: AnnotatedRow[]
  fileHandle?: FileSystemFileHandle
  outputContainer?: OutputContainer
}

export type RenderWorkerResponse =
  | { type: 'progress'; progress: number }
  | { type: 'stage'; stage: RenderStage }
  | { type: 'done'; blob: Blob | null }
  | { type: 'error'; message: string }

self.onmessage = async (event: MessageEvent<RenderWorkerRequest>) => {
  const { videoFile, trimStart, trimEnd, config, annotatedRows, fileHandle, outputContainer } = event.data
  let outputStream: FileSystemWritableFileStream | undefined
  try {
    outputStream = fileHandle ? await fileHandle.createWritable() : undefined
    const blob = await renderVideo(videoFile, {
      trimStart,
      trimEnd,
      config,
      annotatedRows,
      outputContainer,
      outputStream,
      onProgress: (progress) => {
        const message: RenderWorkerResponse = { type: 'progress', progress }
        self.postMessage(message)
      },
      onStage: (stage) => {
        const message: RenderWorkerResponse = { type: 'stage', stage }
        self.postMessage(message)
      },
    })
    // Deliberately *not* closed here. mediabunny's StreamTarget takes a
    // writer on this stream (which locks it) and closes that writer when the
    // Output is finalized -- and for a FileSystemWritableFileStream, that
    // close is what commits the bytes to disk. Closing it a second time here
    // threw "Cannot close a locked stream" at the very end of every
    // save-to-file render, after all the work was already done.
    const message: RenderWorkerResponse = { type: 'done', blob }
    self.postMessage(message)
  } catch (error) {
    // Best-effort: drop the partial file rather than leaving the one the user
    // picked holding a truncated video. Guarded on `locked` because if
    // mediabunny still holds its writer the stream is no longer ours to
    // abort, and throwing here would bury the real error.
    if (outputStream && !outputStream.locked) {
      await outputStream.abort().catch(() => undefined)
    }
    const message: RenderWorkerResponse = { type: 'error', message: error instanceof Error ? error.message : String(error) }
    self.postMessage(message)
  }
}
