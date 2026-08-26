/** Where a rendered video ends up. File System Access streams the encoded
 * output straight to disk as it's produced, so a multi-GB render is never
 * fully buffered in memory; it's Chromium-only, so browsers without it
 * (Firefox, Safari) fall back to buffering the whole thing as a Blob and
 * downloading it once finished -- see pipeline.ts's `outputStream` option
 * for the tradeoff this avoids/accepts either way. */

export function hasFileSystemAccess(): boolean {
  return typeof window !== 'undefined' && 'showSaveFilePicker' in window
}

export async function pickSaveFile(suggestedName: string): Promise<FileSystemFileHandle | null> {
  try {
    return await window.showSaveFilePicker({
      suggestedName,
      types: [{ description: 'MP4 video', accept: { 'video/mp4': ['.mp4'] } }],
    })
  } catch (error) {
    // AbortError -- the user canceled the picker. Anything else, surface it.
    if (error instanceof DOMException && error.name === 'AbortError') return null
    throw error
  }
}
