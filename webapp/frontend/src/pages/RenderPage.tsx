import { useState } from 'react'
import { downloadUrl } from '../api/client'
import { RenderProgress } from '../components/RenderProgress'
import type { JobStatus } from '../types'

interface RenderPageProps {
  jobId: string
  onStartOver: () => void
}

export function RenderPage({ jobId, onStartOver }: RenderPageProps) {
  const [finished, setFinished] = useState<JobStatus | null>(null)

  return (
    <div className="page render-page">
      <h1>Rendering your video</h1>
      <p className="page__subtitle">Trimming, drawing the HUD, and compositing it onto your footage. This can take a while for long clips.</p>

      <RenderProgress jobId={jobId} onDone={setFinished} />

      {finished?.status === 'done' && (
        <a className="primary-button" href={downloadUrl(jobId)} download>
          Download video
        </a>
      )}

      <button className="secondary-button" onClick={onStartOver}>
        Start over
      </button>
    </div>
  )
}
