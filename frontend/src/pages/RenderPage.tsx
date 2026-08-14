import { useEffect, useState } from 'react'
import { downloadUrl, getVideo, releaseJob } from '../api/client'
import { RenderProgress } from '../components/RenderProgress'
import type { JobStatus } from '../types'

interface RenderPageProps {
  jobId: string
  videoId: string
  claimToken: string
  onStartOver: () => void
}

function formatExpiry(expiresAt: string): string {
  const d = new Date(expiresAt)
  const minutesLeft = Math.max(0, Math.round((d.getTime() - Date.now()) / 60000))
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  return `${time} (about ${minutesLeft} min from now)`
}

function copyText(text: string): boolean {
  // navigator.clipboard only exists in "secure contexts" (HTTPS or
  // localhost) -- accessing this app over plain http:// (e.g. its Tailscale
  // IP, which is exactly how this is meant to be reached from another
  // device) means navigator.clipboard is undefined entirely, not just
  // permission-denied, so calling it throws synchronously. execCommand is
  // deprecated but is the only copy mechanism that still works in that
  // situation, so it's used as a fallback rather than the only path.
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).catch(() => {})
    return true
  }
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  let ok = false
  try {
    ok = document.execCommand('copy')
  } catch {
    ok = false
  }
  document.body.removeChild(textarea)
  return ok
}

function SelfRenderSnippet({ jobId, claimToken, released }: { jobId: string; claimToken: string; released: boolean }) {
  const [copied, setCopied] = useState(false)
  const [copyFailed, setCopyFailed] = useState(false)
  const [releasing, setReleasing] = useState(false)
  const command = `docker run --rm --pull always ghcr.io/jlleongarcia/py_routetracker:latest \\\n  python -m app.worker_main --server ${window.location.origin} --job ${jobId} --token ${claimToken}`

  const copy = () => {
    if (copyText(command)) {
      setCopyFailed(false)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } else {
      setCopyFailed(true)
    }
  }

  const renderOnServer = async () => {
    setReleasing(true)
    try {
      await releaseJob(jobId)
    } catch {
      setReleasing(false)
    }
  }

  if (released) {
    return (
      <div className="self-render">
        <p className="self-render__hint">Waiting for the server to pick this up — it'll start shortly.</p>
      </div>
    )
  }

  return (
    <div className="self-render">
      <p className="self-render__title">Want this to render faster? Run it on your own device:</p>
      <pre className="self-render__command">{command}</pre>
      <button className="secondary-button" onClick={copy}>
        {copied ? 'Copied!' : 'Copy command'}
      </button>
      {copyFailed && (
        <p className="self-render__hint">
          Couldn't copy automatically — select the text above and copy it by hand instead.
        </p>
      )}
      <p className="self-render__hint">
        Needs Docker on that device. This waits for you either way — nothing renders until you decide.
      </p>
      <button className="secondary-button" onClick={renderOnServer} disabled={releasing}>
        {releasing ? 'Starting…' : 'Render on the server instead'}
      </button>
    </div>
  )
}

export function RenderPage({ jobId, videoId, claimToken, onStartOver }: RenderPageProps) {
  const [finished, setFinished] = useState<JobStatus | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
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

      <RenderProgress jobId={jobId} onDone={setFinished} onUpdate={setStatus} />

      {status?.status === 'pending' && (
        <SelfRenderSnippet jobId={jobId} claimToken={claimToken} released={status.released} />
      )}

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
