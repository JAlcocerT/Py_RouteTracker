interface PartsListProps {
  parts: File[]
  onReorder: (parts: File[]) => void
  autoSorted: boolean
}

function move(parts: File[], from: number, to: number): File[] {
  const next = [...parts]
  const [moved] = next.splice(from, 1)
  next.splice(to, 0, moved)
  return next
}

export function PartsList({ parts, onReorder, autoSorted }: PartsListProps) {
  if (parts.length === 0) return null

  return (
    <div className="parts-list">
      <div className="parts-list__hint">
        {autoSorted
          ? 'Detected GoPro chapter naming — parts below are auto-ordered. Reorder manually if this is wrong.'
          : 'Arrange these in the order they should play, first to last.'}
      </div>
      <ol className="parts-list__items">
        {parts.map((file, i) => (
          <li key={`${file.name}-${i}`} className="parts-list__item">
            <span className="parts-list__index">{i + 1}</span>
            <span className="parts-list__filename">{file.name}</span>
            <div className="parts-list__actions">
              <button
                type="button"
                disabled={i === 0}
                onClick={() => onReorder(move(parts, i, i - 1))}
                aria-label={`Move ${file.name} earlier`}
              >
                ↑
              </button>
              <button
                type="button"
                disabled={i === parts.length - 1}
                onClick={() => onReorder(move(parts, i, i + 1))}
                aria-label={`Move ${file.name} later`}
              >
                ↓
              </button>
              <button
                type="button"
                onClick={() => onReorder(parts.filter((_, idx) => idx !== i))}
                aria-label={`Remove ${file.name}`}
              >
                ✕
              </button>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
