import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'

interface DropzoneProps {
  label: string
  accept: string
  file: File | null
  onFile: (file: File | null) => void
  hint?: string
}

export function Dropzone({ label, accept, file, onFile, hint }: DropzoneProps) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragging(false)
      const dropped = e.dataTransfer.files?.[0]
      if (dropped) onFile(dropped)
    },
    [onFile],
  )

  return (
    <div
      className={`dropzone ${dragging ? 'dropzone--active' : ''} ${file ? 'dropzone--filled' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <>
          <div className="dropzone__icon">✓</div>
          <div className="dropzone__filename">{file.name}</div>
          <div className="dropzone__hint">click to replace</div>
        </>
      ) : (
        <>
          <div className="dropzone__icon">⇪</div>
          <div className="dropzone__label">{label}</div>
          {hint && <div className="dropzone__hint">{hint}</div>}
        </>
      )}
    </div>
  )
}
