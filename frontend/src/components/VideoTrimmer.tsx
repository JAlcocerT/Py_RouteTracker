import { useRef } from 'react'

interface VideoTrimmerProps {
  videoUrl: string
  duration: number
  trimStart: number
  trimEnd: number
  onChange: (start: number, end: number) => void
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

export function VideoTrimmer({ videoUrl, duration, trimStart, trimEnd, onChange }: VideoTrimmerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  const seekTo = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t
  }

  return (
    <div className="video-trimmer">
      <video ref={videoRef} src={videoUrl} controls className="video-trimmer__player" />
      <div className="video-trimmer__range">
        <div
          className="video-trimmer__selection"
          style={{
            left: `${(trimStart / duration) * 100}%`,
            width: `${((trimEnd - trimStart) / duration) * 100}%`,
          }}
        />
        <input
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={trimStart}
          onChange={(e) => {
            const v = Math.min(Number(e.target.value), trimEnd - 0.5)
            onChange(v, trimEnd)
            seekTo(v)
          }}
          className="video-trimmer__handle video-trimmer__handle--start"
        />
        <input
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={trimEnd}
          onChange={(e) => {
            const v = Math.max(Number(e.target.value), trimStart + 0.5)
            onChange(trimStart, v)
            seekTo(v)
          }}
          className="video-trimmer__handle video-trimmer__handle--end"
        />
      </div>
      <div className="video-trimmer__labels">
        <span>{formatTime(trimStart)}</span>
        <span className="video-trimmer__duration">selected: {formatTime(trimEnd - trimStart)}</span>
        <span>{formatTime(trimEnd)}</span>
      </div>
    </div>
  )
}
