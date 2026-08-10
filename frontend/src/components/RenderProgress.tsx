import { useEffect, useState } from 'react'
import { getJob } from '../api/client'
import type { JobStatus } from '../types'

interface RenderProgressProps {
  jobId: string
  onDone?: (job: JobStatus) => void
}

export function RenderProgress({ jobId, onDone }: RenderProgressProps) {
  const [job, setJob] = useState<JobStatus | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number

    const poll = async () => {
      try {
        const status = await getJob(jobId)
        if (cancelled) return
        setJob(status)
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
  }, [jobId, onDone])

  if (!job) return <div className="render-progress">Starting…</div>

  const pct = Math.round(job.progress * 100)

  return (
    <div className="render-progress">
      <div className="render-progress__bar">
        <div className="render-progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="render-progress__label">
        {job.status === 'running' && `Rendering… ${pct}%`}
        {job.status === 'pending' && 'Queued…'}
        {job.status === 'done' && 'Done!'}
        {job.status === 'error' && `Failed: ${job.error}`}
      </div>
    </div>
  )
}
