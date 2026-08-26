import { useState } from 'react'
import { RenderProgress, type RenderPhase } from '../components/RenderProgress'
import type { AnnotatedRow } from '../lib/laps/detection'
import { hasFileSystemAccess, pickSaveFile } from '../lib/output'
import { hudConfigFor } from '../lib/render/renderConfig'
import { runRender } from '../lib/render/runRender'
import type { RenderStyle, WidgetSelection } from '../types'

interface RenderPageProps {
  videoFile: File
  trimStart: number
  trimEnd: number
  widgets: WidgetSelection
  style: RenderStyle
  annotatedRows: AnnotatedRow[]
  onStartOver: () => void
}

function outputName(videoFile: File): string {
  const base = videoFile.name.replace(/\.[^.]+$/, '')
  return `${base}_overlay.mp4`
}

export function RenderPage({ videoFile, trimStart, trimEnd, widgets, style, annotatedRows, onStartOver }: RenderPageProps) {
  const [phase, setPhase] = useState<RenderPhase>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [savedToDisk, setSavedToDisk] = useState(false)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)

  const handleStart = async () => {
    setPhase('rendering')
    setProgress(0)
    setError(null)
    try {
      // The save-file picker needs a user gesture, so it's opened here, at
      // the very start of this click handler, rather than being deferred
      // until the render finishes -- see lib/output.ts.
      const fileHandle = hasFileSystemAccess() ? await pickSaveFile(outputName(videoFile)) : null

      const blob = await runRender({
        videoFile,
        trimStart,
        trimEnd,
        config: hudConfigFor(widgets, style),
        annotatedRows,
        fileHandle: fileHandle ?? undefined,
        onProgress: setProgress,
      })

      if (blob) {
        setDownloadUrl(URL.createObjectURL(blob))
      } else {
        setSavedToDisk(true)
      }
      setPhase('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('error')
    }
  }

  return (
    <div className="page render-page">
      <h1>Rendering your video</h1>
      <p className="page__subtitle">Trimming, drawing the HUD, and compositing it onto your footage -- entirely in this browser tab. This can take a while for long clips.</p>

      {phase === 'idle' && (
        <button className="primary-button" onClick={handleStart}>
          Start rendering
        </button>
      )}

      {phase !== 'idle' && <RenderProgress phase={phase} progress={progress} error={error} />}

      {phase === 'done' && (
        <>
          {savedToDisk ? (
            <p className="expiry-notice">Saved to the location you chose.</p>
          ) : (
            downloadUrl && (
              <a className="primary-button" href={downloadUrl} download={outputName(videoFile)}>
                Download video
              </a>
            )
          )}
        </>
      )}

      <button className="secondary-button" onClick={onStartOver}>
        Start over
      </button>
    </div>
  )
}
