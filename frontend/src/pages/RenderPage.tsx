import { useEffect, useState } from 'react'
import { downloadUrl, getVideo } from '../api/client'
import { RenderProgress } from '../components/RenderProgress'
import type { JobStatus } from '../types'

interface RenderPageProps {
  jobId: string
  videoId: string
  onStartOver: () => void
}

function formatExpiry(expiresAt: string): string {
  const d = new Date(expiresAt)
  const minutesLeft = Math.max(0, Math.round((d.getTime() - Date.now()) / 60000))
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  return `${time} (about ${minutesLeft} min from now)`
}

export function RenderPage({ jobId, videoId, onStartOver }: RenderPageProps) {
  const [finished, setFinished] = useState<JobStatus | null>(null)
  const [expiresAt, setExpiresAt] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getVideo(videoId)
      .then((meta) => {
        if (!cancelled) setExpiresAt(meta.expires_at)
      })
      .catch(() => {
        // non-critical -- the download still works even if we can't show the expiry notice
      })
    return () => {
      cancelled = true
    }
  }, [videoId])

  return (
    <div className="page render-page">
      <h1>Rendering your video</h1>
      <p className="page__subtitle">Trimming, drawing the HUD, and compositing it onto your footage. This can take a while for long clips.</p>

      <RenderProgress jobId={jobId} onDone={setFinished} />

      {finished?.status === 'done' && (
        <>
          <a className="primary-button" href={downloadUrl(jobId)} download>
            Download video
          </a>
          <p className="expiry-notice">
            Uploaded videos and rendered output are temporary — nothing is kept in permanent
            storage.{' '}
            {expiresAt ? (
              <>This one will be automatically deleted around <strong>{formatExpiry(expiresAt)}</strong>, so download it now.</>
            ) : (
              <>Download it now, before it is automatically deleted.</>
            )}
          </p>
        </>
      )}

      <button className="secondary-button" onClick={onStartOver}>
        Start over
      </button>
    </div>
  )
}
