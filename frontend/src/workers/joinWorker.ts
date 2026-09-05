/** Hosts the lossless part-join (see lib/mp4/join.ts) off the main thread,
 * for the same reason renderWorker.ts exists: the authoring half of a join
 * is a synchronous walk over every sample of every part, and on the main
 * thread it froze the tab -- and with it the progress bar the user is
 * watching -- for as long as it took. */
import { joinVideos } from '../lib/mp4/join'

export interface JoinWorkerRequest {
  parts: File[]
}

export type JoinWorkerResponse =
  | { type: 'progress'; progress: number }
  | { type: 'done'; blob: Blob }
  | { type: 'error'; message: string }

self.onmessage = async (event: MessageEvent<JoinWorkerRequest>) => {
  try {
    const blob = await joinVideos(event.data.parts, (progress) => {
      const message: JoinWorkerResponse = { type: 'progress', progress }
      self.postMessage(message)
    })
    const message: JoinWorkerResponse = { type: 'done', blob }
    self.postMessage(message)
  } catch (error) {
    const message: JoinWorkerResponse = {
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    }
    self.postMessage(message)
  }
}
