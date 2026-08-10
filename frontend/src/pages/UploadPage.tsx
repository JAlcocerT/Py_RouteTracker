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
  const [busy, setBusy] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [error, setError] = useState<string | null>(null)

  const canSubmit = video != null && (sourceType === 'gopro_embedded' || gpx != null) && !busy

  const handleSubmit = async () => {
    if (!video) return
    setBusy(true)
    setError(null)
    try {
      setStatusText('Uploading…')
      const { video_id, job_id } = await uploadVideo({
        video,
        sourceType,
        gpx: gpx ?? undefined,
        videoStartTime: videoStartTime ? new Date(videoStartTime).toISOString() : undefined,
      })

      setStatusText('Extracting telemetry…')
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
      setBusy(false)
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

      <button className="primary-button" disabled={!canSubmit} onClick={handleSubmit}>
        {busy ? statusText : 'Extract telemetry'}
      </button>
    </div>
  )
}
