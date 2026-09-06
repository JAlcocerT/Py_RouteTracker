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
  const softwareDecoding = stage === 'software-decoding'

  return (
    <div className="progress">
      <div className="progress__bar">
        <div className="progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress__label">
        {phase === 'idle' && 'Ready to render.'}
        {phase === 'rendering' && `Rendering${softwareDecoding ? ' (software decoding)' : ''}… ${pct}%`}
        {phase === 'done' && 'Done!'}
        {phase === 'error' && `Failed: ${error}`}
      </div>
      {phase === 'rendering' && softwareDecoding && (
        // Without this the render just looks hung: it can run for many
        // minutes with the bar barely moving, since decoding HEVC in software
        // is far slower than the hardware path it's standing in for.
        <p className="progress__note">
          This browser can't decode HEVC video itself, so it's being decoded in software. The render still produces the same
          result, but it's much slower and will use a lot of CPU — leave this tab open.
        </p>
      )}
    </div>
  )
}
