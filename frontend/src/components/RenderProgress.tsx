export type RenderPhase = 'idle' | 'rendering' | 'done' | 'error'

interface RenderProgressProps {
  phase: RenderPhase
  progress: number
  error: string | null
}

export function RenderProgress({ phase, progress, error }: RenderProgressProps) {
  const pct = Math.round(progress * 100)

  return (
    <div className="progress">
      <div className="progress__bar">
        <div className="progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress__label">
        {phase === 'idle' && 'Ready to render.'}
        {phase === 'rendering' && `Rendering… ${pct}%`}
        {phase === 'done' && 'Done!'}
        {phase === 'error' && `Failed: ${error}`}
      </div>
    </div>
  )
}
