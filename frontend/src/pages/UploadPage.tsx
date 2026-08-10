import { useState } from 'react'
import { getJob, uploadVideo } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import type { SourceType } from '../types'

interface UploadPageProps {
  onUploaded: (videoId: string, videoFile: File) => void
}

export function UploadPage({ onUploaded }: UploadPageProps) {
  const [sourceType, setSourceType] = useState<SourceType>('gopro_embedded')
  const [video, setVideo] = useState<File | null>(null)
  const [gpx, setGpx] = useState<File | null>(null)
  const [videoStartTime, setVideoStartTime] = useState('')
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'extracting'>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const busy = phase !== 'idle'
  const canSubmit = video != null && (sourceType === 'gopro_embedded' || gpx != null) && !busy

  const handleSubmit = async () => {
    if (!video) return
    setPhase('uploading')
    setUploadProgress(0)
    setError(null)
    try {
      const { video_id, job_id } = await uploadVideo({
        video,
        sourceType,
        gpx: gpx ?? undefined,
        videoStartTime: videoStartTime ? new Date(videoStartTime).toISOString() : undefined,
        onProgress: setUploadProgress,
      })

      setPhase('extracting')
      // poll until the background extraction job finishes
      for (;;) {
        const job = await getJob(job_id)
        if (job.status === 'done') break
        if (job.status === 'error') throw new Error(job.error ?? 'extraction failed')
        await new Promise((r) => setTimeout(r, 600))
      }

      onUploaded(video_id, video)
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
        <button className={sourceType === 'gopro_embedded' ? 'active' : ''} onClick={() => setSourceType('gopro_embedded')}>
          GoPro embedded GPS
        </button>
        <button className={sourceType === 'external_gpx' ? 'active' : ''} onClick={() => setSourceType('external_gpx')}>
          Video + separate GPX file
        </button>
      </div>

      <Dropzone
        label="Drop your video here"
        accept="video/*"
        file={video}
        onFile={setVideo}
        hint={sourceType === 'gopro_embedded' ? 'GoPro MP4 with embedded GPS/GPMD metadata' : 'any action-cam video'}
      />

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
        {phase === 'uploading' && 'Uploading…'}
        {phase === 'extracting' && 'Extracting telemetry…'}
        {phase === 'idle' && 'Extract telemetry'}
      </button>
    </div>
  )
}
