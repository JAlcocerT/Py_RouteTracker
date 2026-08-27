import { useEffect, useState } from 'react'
import { Dropzone } from '../components/Dropzone'
import { MultiDropzone } from '../components/MultiDropzone'
import { PartsList } from '../components/PartsList'
import { describeCodecCompatIssue } from '../lib/env'
import { probeVideoDuration } from '../lib/mp4/probe'
import { joinVideos } from '../lib/mp4/join'
import { extractExternalGpx } from '../lib/telemetry/externalGpx'
import { extractGoProGpmf } from '../lib/telemetry/goproGpmf'
import type { SourceType, TelemetryPoint } from '../types'
import { autoSortGoProParts } from '../utils/videoParts'

interface UploadPageProps {
  onUploaded: (videoFile: File, duration: number, telemetry: TelemetryPoint[], hasAccel: boolean) => void
}

type UploadMode = 'single' | 'join'

export function UploadPage({ onUploaded }: UploadPageProps) {
  const [mode, setMode] = useState<UploadMode>('single')
  const [sourceType, setSourceType] = useState<SourceType>('gopro_embedded')
  const [video, setVideo] = useState<File | null>(null)
  const [videoParts, setVideoParts] = useState<File[]>([])
  const [partsAutoSorted, setPartsAutoSorted] = useState(false)
  const [gpx, setGpx] = useState<File | null>(null)
  const [videoStartTime, setVideoStartTime] = useState('')
  const [phase, setPhase] = useState<'idle' | 'joining' | 'extracting'>('idle')
  const [joinProgress, setJoinProgress] = useState(0)
  const [extractProgress, setExtractProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [compatWarning, setCompatWarning] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    describeCodecCompatIssue().then((message) => {
      if (!cancelled) setCompatWarning(message)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const busy = phase !== 'idle'
  const hasVideo = mode === 'single' ? video != null : videoParts.length >= 2
  const canSubmit = hasVideo && (sourceType === 'gopro_embedded' || gpx != null) && !busy

  const addParts = (files: File[]) => {
    const merged = [...videoParts, ...files]
    const sorted = autoSortGoProParts(merged)
    setVideoParts(sorted ?? merged)
    setPartsAutoSorted(sorted != null)
  }

  const reorderParts = (next: File[]) => {
    setVideoParts(next)
    setPartsAutoSorted(false) // any manual change overrides the auto-sort hint
  }

  const handleSubmit = async () => {
    setError(null)
    try {
      let videoFile: File

      if (mode === 'single') {
        if (!video) return
        videoFile = video
      } else {
        setPhase('joining')
        setJoinProgress(0)
        const blob = await joinVideos(videoParts, setJoinProgress)
        videoFile = new File([blob], videoParts[0].name, { type: 'video/mp4' })
      }

      setPhase('extracting')
      setExtractProgress(0)
      const duration = await probeVideoDuration(videoFile)

      const result =
        sourceType === 'gopro_embedded'
          ? await extractGoProGpmf(videoFile, duration, { onProgress: setExtractProgress })
          : extractExternalGpx(await gpx!.text(), duration, {
              videoStartTime: videoStartTime ? new Date(videoStartTime) : undefined,
            })

      if (result.rows.length === 0) {
        throw new Error(
          sourceType === 'gopro_embedded'
            ? 'No embedded GPS telemetry found in this video.'
            : "The GPX track doesn't overlap the video's timeline -- check the start time.",
        )
      }

      onUploaded(videoFile, duration, result.rows, result.hasAccel)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('idle')
    }
  }

  return (
    <div className="page upload-page">
      <h1>Telemetry Overlay</h1>
      <p className="page__subtitle">Drop in a video, pick your telemetry source, and we'll extract GPS + speed data -- entirely in this browser tab, nothing uploaded anywhere.</p>

      {compatWarning && <p className="compat-banner">{compatWarning}</p>}

      <div className="source-toggle">
        <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')} disabled={busy}>
          Single video file
        </button>
        <button className={mode === 'join' ? 'active' : ''} onClick={() => setMode('join')} disabled={busy}>
          Join split recording (multiple parts)
        </button>
      </div>

      <div className="source-toggle">
        <button className={sourceType === 'gopro_embedded' ? 'active' : ''} onClick={() => setSourceType('gopro_embedded')}>
          GoPro embedded GPS
        </button>
        <button className={sourceType === 'external_gpx' ? 'active' : ''} onClick={() => setSourceType('external_gpx')}>
          Video + separate GPX file
        </button>
      </div>

      {mode === 'single' ? (
        <Dropzone
          label="Drop your video here"
          accept="video/*"
          file={video}
          onFile={setVideo}
          hint={sourceType === 'gopro_embedded' ? 'GoPro MP4 with embedded GPS/GPMD metadata' : 'any action-cam video'}
        />
      ) : (
        <>
          <MultiDropzone
            label="Drop all parts of the split recording here"
            accept="video/*"
            hint="e.g. GH010437.MP4 + GH020437.MP4 — chapters of the same recording, same camera/codec/resolution"
            onFiles={addParts}
          />
          <PartsList parts={videoParts} onReorder={reorderParts} autoSorted={partsAutoSorted} />
        </>
      )}

      {sourceType === 'external_gpx' && (
        <>
          <Dropzone label="Drop your GPX track here" accept=".gpx" file={gpx} onFile={setGpx} />
          <label className="field">
            <span>Video start time (optional — helps sync the GPX track to the footage)</span>
            <input type="datetime-local" step={1} value={videoStartTime} onChange={(e) => setVideoStartTime(e.target.value)} />
          </label>
        </>
      )}

      {error && <p className="error-text">{error}</p>}

      {phase === 'joining' && (
        <div className="progress">
          <div className="progress__bar">
            <div className="progress__fill" style={{ width: `${Math.round(joinProgress * 100)}%` }} />
          </div>
          <div className="progress__label">Joining parts… {Math.round(joinProgress * 100)}%</div>
        </div>
      )}

      {phase === 'extracting' && sourceType === 'gopro_embedded' && (
        <div className="progress">
          <div className="progress__bar">
            <div className="progress__fill" style={{ width: `${Math.round(extractProgress * 100)}%` }} />
          </div>
          <div className="progress__label">Extracting telemetry… {Math.round(extractProgress * 100)}%</div>
        </div>
      )}

      <button className="primary-button" disabled={!canSubmit} onClick={handleSubmit}>
        {phase === 'joining' && 'Joining…'}
        {phase === 'extracting' && 'Extracting telemetry…'}
        {phase === 'idle' && 'Extract telemetry'}
      </button>
    </div>
  )
}
