import type { RenderStage } from '../lib/render/pipeline'

export type RenderPhase = 'idle' | 'rendering' | 'done' | 'error'

interface RenderProgressProps {
  phase: RenderPhase
  progress: number
  error: string | null
  stage?: RenderStage
}

export function RenderProgress({ phase, progress, error, stage = 'rendering' }: RenderProgressProps) {
  const pct = Math.round(progress * 100)
  const transcoding = stage === 'transcoding'

  return (
    <div className="progress">
      <div className="progress__bar">
        <div className="progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress__label">
        {phase === 'idle' && 'Ready to render.'}
        {phase === 'rendering' && `${transcoding ? 'Converting video' : 'Rendering'}… ${pct}%`}
        {phase === 'done' && 'Done!'}
        {phase === 'error' && `Failed: ${error}`}
      </div>
      {phase === 'rendering' && transcoding && (
        // Without this the fallback just looks like a hung render: it can
        // run for many minutes with the bar barely moving, since software
        // decoding is far slower than the hardware path it's standing in for.
        <p className="progress__note">
          This browser can't decode your footage's format directly, so it's being converted first. This is much slower than
          normal and will use a lot of CPU — leave this tab open.
        </p>
      )}
    </div>
  )
}
