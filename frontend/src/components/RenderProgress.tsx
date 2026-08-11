import { useEffect, useState } from 'react'
import { getJob } from '../api/client'
import type { JobStatus } from '../types'

interface RenderProgressProps {
  jobId: string
  onDone?: (job: JobStatus) => void
  /** Fired on every poll, including non-terminal statuses -- lets a parent
   * react to e.g. "still pending" (see RenderPage's self-render snippet). */
  onUpdate?: (job: JobStatus) => void
}

export function RenderProgress({ jobId, onDone, onUpdate }: RenderProgressProps) {
  const [job, setJob] = useState<JobStatus | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number

    const poll = async () => {
      try {
        const status = await getJob(jobId)
        if (cancelled) return
        setJob(status)
        onUpdate?.(status)
        if (status.status === 'done' || status.status === 'error') {
          onDone?.(status)
          return
        }
      } catch {
        // transient network hiccup -- keep polling
      }
      timer = window.setTimeout(poll, 700)
    }
    poll()

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [jobId, onDone, onUpdate])

  if (!job) return <div className="progress">Starting…</div>

  const pct = Math.round(job.progress * 100)

  return (
    <div className="progress">
      <div className="progress__bar">
        <div className="progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress__label">
        {job.status === 'running' && `Rendering… ${pct}%`}
        {job.status === 'pending' && 'Queued…'}
        {job.status === 'done' && 'Done!'}
        {job.status === 'error' && `Failed: ${job.error}`}
        {job.worker_id && job.worker_id !== 'local' && (job.status === 'running' || job.status === 'done') && (
          <span className="progress__worker"> — rendered on {job.worker_id === 'self' ? 'your own device' : job.worker_id}</span>
        )}
      </div>
    </div>
  )
}
