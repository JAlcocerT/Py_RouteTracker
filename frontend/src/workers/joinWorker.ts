/** Hosts the lossless part-join (see lib/mp4/join.ts) off the main thread,
 * for the same reason renderWorker.ts exists: the authoring half of a join
 * is a synchronous walk over every sample of every part, and on the main
 * thread it froze the tab -- and with it the progress bar the user is
 * watching -- for as long as it took.
 *
 * It also owns where the joined bytes land. A joined recording is routinely
 * bigger than the 4 GiB a single ArrayBuffer can hold, so it's streamed to a
 * scratch file in the origin-private filesystem and handed back as a File
 * over that, rather than buffered. OPFS rather than showSaveFilePicker
 * because this file is an intermediate the user never asked for: it exists
 * only to be probed and rendered, so it needs no prompt, no user gesture,
 * and no place on their disk. */
import { type JoinVideosOptions, joinVideos } from '../lib/mp4/join'

export interface JoinWorkerRequest {
  parts: File[]
}

/** A joined recording as the rest of the app consumes it: an ordinary File,
 * whether it was streamed to OPFS or buffered. */
export interface JoinedRecording {
  file: File
  /** Where each input part begins in the joined timeline, in seconds. */
  partStarts: number[]
}

export type JoinWorkerResponse =
  | { type: 'progress'; progress: number }
  | { type: 'done'; joined: JoinedRecording }
  | { type: 'error'; message: string }

/** Everything a join writes lives here, so the previous one can be cleared
 * wholesale without tracking names. */
const SCRATCH_DIR = 'joins'

const gb = (bytes: number) => (bytes / 1e9).toFixed(1)

/** OPFS draws on the origin's storage quota, and a joined recording is the
 * size of every part put together. Checking up front turns what would
 * otherwise be a failure minutes into a join into an immediate, explainable
 * one. Silent where the browser won't estimate: better to attempt the join
 * than to refuse it on a number we don't have. */
async function checkQuota(needed: number): Promise<void> {
  const estimate: StorageEstimate = (await navigator.storage.estimate?.().catch(() => ({}))) ?? {}
  const { quota, usage } = estimate
  if (quota === undefined || usage === undefined || quota - usage >= needed) return
  throw new Error(
    `Not enough browser storage to join these parts. They need about ${gb(needed)} GB of scratch ` +
      `space and only ${gb(quota - usage)} GB is free. Clear some disk space, or this site's stored ` +
      'data, and try again.',
  )
}

/** An empty OPFS file to stream the join into, or null where OPFS isn't
 * available to write (Safari before 17 exposes handles but no createWritable)
 * -- callers fall back to buffering there. */
async function openScratchFile(name: string): Promise<FileSystemFileHandle | null> {
  if (!navigator.storage?.getDirectory) return null
  const root = await navigator.storage.getDirectory()
  // Drop what the last join left behind before writing a new one. These are
  // whole-recording sized and nothing reads them past the upload they belong
  // to, so keeping them around would quietly fill the origin's quota.
  await root.removeEntry(SCRATCH_DIR, { recursive: true }).catch(() => undefined)
  const dir = await root.getDirectoryHandle(SCRATCH_DIR, { create: true })
  const handle = await dir.getFileHandle(name, { create: true })
  return 'createWritable' in handle ? handle : null
}

self.onmessage = async (event: MessageEvent<JoinWorkerRequest>) => {
  const { parts } = event.data
  let outputStream: FileSystemWritableFileStream | undefined
  try {
    const name = parts[0]?.name ?? 'joined.mp4'
    // Named after the first part, because getFile() below takes the File's
    // name from the handle and the rest of the app shows it to the user.
    const handle = await openScratchFile(name).catch(() => null)
    if (handle) await checkQuota(parts.reduce((total, part) => total + part.size, 0))
    outputStream = await handle?.createWritable()

    const { blob, partStarts } = await joinVideos(parts, {
      outputStream,
      onProgress: (progress) => {
        const message: JoinWorkerResponse = { type: 'progress', progress }
        self.postMessage(message)
      },
    } satisfies JoinVideosOptions)

    // Deliberately *not* closing outputStream here -- mediabunny's
    // StreamTarget holds a writer on it and closes that on finalize, and for
    // a FileSystemWritableFileStream that close is what commits the bytes.
    // Closing twice throws "Cannot close a locked stream" (see renderWorker).
    const file = blob ? new File([blob], name, { type: 'video/mp4' }) : await handle!.getFile()
    const message: JoinWorkerResponse = { type: 'done', joined: { file, partStarts } }
    self.postMessage(message)
  } catch (error) {
    // Best-effort: drop the half-written scratch file rather than leaving a
    // truncated video behind. Guarded on `locked` because once mediabunny has
    // its writer the stream isn't ours to abort, and throwing here would bury
    // the real error.
    if (outputStream && !outputStream.locked) {
      await outputStream.abort().catch(() => undefined)
    }
    // A quota blowout part way through says only "QuotaExceededError" on its
    // own, which tells the user nothing about what to do next.
    const quotaExceeded = error instanceof DOMException && error.name === 'QuotaExceededError'
    const message: JoinWorkerResponse = {
      type: 'error',
      message: quotaExceeded
        ? "Ran out of browser storage part way through the join. Clear some disk space, or this site's " +
          'stored data, and try again.'
        : error instanceof Error
          ? error.message
          : String(error),
    }
    self.postMessage(message)
  }
}
