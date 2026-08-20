// GoPro's modern chaptered-recording naming: GH/GX + 2-digit chapter number
// + 4-digit file number (e.g. GH010437.MP4 is chapter 1, GH020437.MP4
// chapter 2, of recording 0437). Only the two-letter prefix + digit
// grouping is camera-model-specific; everything else is a plain sequence.
const GOPRO_PART_RE = /^(?:GH|GX)(\d{2})(\d{4})\./i

interface GoProPartInfo {
  chapter: number
  fileNumber: string
}

function parseGoProPart(filename: string): GoProPartInfo | null {
  const m = GOPRO_PART_RE.exec(filename)
  if (!m) return null
  return { chapter: Number(m[1]), fileNumber: m[2] }
}

/**
 * Returns a chapter-ordered copy of `files` when every one of them matches
 * GoPro's GHccNNNN naming *and* shares the same NNNN (proof they're
 * chapters of the same recording, not unrelated clips someone dropped
 * together) -- otherwise returns null so the caller falls back to manual
 * ordering instead of guessing at a sequence it can't actually detect.
 */
export function autoSortGoProParts(files: File[]): File[] | null {
  if (files.length < 2) return null

  const infos = files.map((f) => parseGoProPart(f.name))
  if (infos.some((info) => info === null)) return null

  const fileNumbers = new Set(infos.map((info) => info!.fileNumber))
  if (fileNumbers.size !== 1) return null

  return files
    .map((file, i) => ({ file, chapter: infos[i]!.chapter }))
    .sort((a, b) => a.chapter - b.chapter)
    .map((entry) => entry.file)
}
