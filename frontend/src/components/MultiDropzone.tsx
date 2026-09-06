import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'

interface MultiDropzoneProps {
  label: string
  accept: string
  hint?: string
  onFiles: (files: File[]) => void
}

export function MultiDropzone({ label, accept, hint, onFiles }: MultiDropzoneProps) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragging(false)
      if (e.dataTransfer.files?.length) onFiles(Array.from(e.dataTransfer.files))
    },
    [onFiles],
  )

  return (
    <div
      className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
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
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files))
          e.target.value = ''
        }}
      />
      <div className="dropzone__icon">⇪</div>
      <div className="dropzone__label">{label}</div>
      {hint && <div className="dropzone__hint">{hint}</div>}
    </div>
  )
}
