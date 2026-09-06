import { useEffect, useRef, useState } from 'react'

interface VideoTrimmerProps {
  videoUrl: string
  duration: number
  trimStart: number
  trimEnd: number
  onChange: (start: number, end: number) => void
}

/** A <video> that can't handle the file's codec usually fires `error`, but
 * some builds instead stall forever at readyState 0 without firing anything.
 * Treat that silence as a failed preview too, rather than leaving the user
 * staring at a dead player wondering whether it's still loading. */
const PREVIEW_TIMEOUT_MS = 10_000

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

/** Parses the "M:SS.s" shown in the manual entry fields back into seconds.
 * Also accepts a plain seconds value, since that's what people type when the
 * clip is under a minute. Returns null for anything unparseable. */
function parseTime(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parts = trimmed.split(':')
  if (parts.length > 2) return null
  const nums = parts.map(Number)
  if (nums.some((n) => !Number.isFinite(n) || n < 0)) return null
  return parts.length === 2 ? nums[0] * 60 + nums[1] : nums[0]
}

export function VideoTrimmer({ videoUrl, duration, trimStart, trimEnd, onChange }: VideoTrimmerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  // The preview is judged by what the <video> element actually does, not by
  // probing WebCodecs -- the two don't always agree on codec support, and
  // gating on the wrong one would hide a preview that works fine.
  const [previewBroken, setPreviewBroken] = useState(false)

  useEffect(() => {
    setPreviewBroken(false)
    const timer = setTimeout(() => {
      if (videoRef.current?.readyState === 0) setPreviewBroken(true)
    }, PREVIEW_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [videoUrl])

  const seekTo = (t: number) => {
    const video = videoRef.current
    if (!video || previewBroken) return
    try {
      video.currentTime = t
    } catch {
      // Seeking an element that failed to load throws in some browsers --
      // the trim value itself is already committed by the caller.
    }
  }

  const setStart = (v: number) => {
    const clamped = Math.max(0, Math.min(v, trimEnd - 0.5))
    onChange(clamped, trimEnd)
    seekTo(clamped)
  }

  const setEnd = (v: number) => {
    const clamped = Math.min(duration, Math.max(v, trimStart + 0.5))
    onChange(trimStart, clamped)
    seekTo(clamped)
  }

  return (
    <div className="video-trimmer">
      {previewBroken ? (
        <div className="video-trimmer__no-preview">
          <strong>Preview unavailable in this browser.</strong>
          <span>
            It can't play this file's video codec (commonly HEVC/H.265 from a GoPro). Trimming still works — use the
            speed graph below to find the section you want, and the time fields for exact values.
          </span>
        </div>
      ) : (
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="video-trimmer__player"
          onError={() => setPreviewBroken(true)}
        />
      )}

      <div className="video-trimmer__range">
        <div
          className="video-trimmer__selection"
          style={{
            left: `${(trimStart / duration) * 100}%`,
            width: `${((trimEnd - trimStart) / duration) * 100}%`,
          }}
        />
        <input
          id="trim-start"
          name="trim-start"
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={trimStart}
          onChange={(e) => setStart(Number(e.target.value))}
          className="video-trimmer__handle video-trimmer__handle--start"
        />
        <input
          id="trim-end"
          name="trim-end"
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={trimEnd}
          onChange={(e) => setEnd(Number(e.target.value))}
          className="video-trimmer__handle video-trimmer__handle--end"
        />
      </div>

      <div className="video-trimmer__labels">
        <span>{formatTime(trimStart)}</span>
        <span className="video-trimmer__duration">selected: {formatTime(trimEnd - trimStart)}</span>
        <span>{formatTime(trimEnd)}</span>
      </div>

      {/* Without a preview the slider is the only handle on a potentially
          30-minute recording, where one pixel is several seconds -- these
          give the precision the slider can't. */}
      {previewBroken && (
        <div className="video-trimmer__manual">
          <label className="style-field">
            <span>Start (m:ss)</span>
            <input
              id="trim-start-manual"
              name="trim-start-manual"
              type="text"
              inputMode="decimal"
              defaultValue={formatTime(trimStart)}
              key={`start-${trimStart}`}
              onBlur={(e) => {
                const parsed = parseTime(e.target.value)
                if (parsed != null) setStart(parsed)
              }}
            />
          </label>
          <label className="style-field">
            <span>End (m:ss)</span>
            <input
              id="trim-end-manual"
              name="trim-end-manual"
              type="text"
              inputMode="decimal"
              defaultValue={formatTime(trimEnd)}
              key={`end-${trimEnd}`}
              onBlur={(e) => {
                const parsed = parseTime(e.target.value)
                if (parsed != null) setEnd(parsed)
              }}
            />
          </label>
        </div>
      )}
    </div>
  )
}
