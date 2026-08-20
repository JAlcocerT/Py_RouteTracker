import { useState } from 'react'
import { fetchVideoSourceFile, getJob, joinAndUploadVideo, uploadVideo } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { MultiDropzone } from '../components/MultiDropzone'
import { PartsList } from '../components/PartsList'
import type { SourceType } from '../types'
import { autoSortGoProParts } from '../utils/videoParts'

interface UploadPageProps {
  onUploaded: (videoId: string, videoFile: File) => void
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
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'extracting' | 'fetching_preview'>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

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
    setPhase('uploading')
    setUploadProgress(0)
    setError(null)
    try {
      const startTime = videoStartTime ? new Date(videoStartTime).toISOString() : undefined
      let videoId: string
      let jobId: string
      let previewFile: File

      if (mode === 'single') {
        if (!video) return
        const result = await uploadVideo({ video, sourceType, gpx: gpx ?? undefined, videoStartTime: startTime, onProgress: setUploadProgress })
        videoId = result.video_id
        jobId = result.job_id
        previewFile = video
      } else {
        const result = await joinAndUploadVideo({ videoParts, sourceType, gpx: gpx ?? undefined, videoStartTime: startTime, onProgress: setUploadProgress })
        videoId = result.video_id
        jobId = result.job_id
        setPhase('fetching_preview')
        // The joined file only exists server-side (raw MP4 parts can't be
        // concatenated into a playable blob in the browser) -- fetch it
        // back once so the trim/preview UI has real, playable footage.
        previewFile = await fetchVideoSourceFile(videoId, videoParts[0].name)
      }

      setPhase('extracting')
      // poll until the background extraction job finishes
      for (;;) {
        const job = await getJob(jobId)
        if (job.status === 'done') break
        if (job.status === 'error') throw new Error(job.error ?? 'extraction failed')
        await new Promise((r) => setTimeout(r, 600))
      }

      onUploaded(videoId, previewFile)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('idle')
    }
  }

  return (
    <div className="page upload-page">
      <h1>Telemetry Overlay</h1>
      <p className="page__subtitle">Drop in a video, pick your telemetry source, and we'll extract GPS + speed data.</p>

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

      {phase === 'uploading' && (
        <div className="progress">
          <div className="progress__bar">
            <div className="progress__fill" style={{ width: `${Math.round(uploadProgress * 100)}%` }} />
          </div>
          <div className="progress__label">Uploading… {Math.round(uploadProgress * 100)}%</div>
        </div>
      )}

      <button className="primary-button" disabled={!canSubmit} onClick={handleSubmit}>
        {phase === 'uploading' && (mode === 'join' ? 'Uploading & joining…' : 'Uploading…')}
        {phase === 'fetching_preview' && 'Preparing preview…'}
        {phase === 'extracting' && 'Extracting telemetry…'}
        {phase === 'idle' && 'Extract telemetry'}
      </button>
    </div>
  )
}
