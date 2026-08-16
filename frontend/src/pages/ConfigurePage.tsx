import { useEffect, useMemo, useState } from 'react'
import { detectLaps, getTelemetry, getVideo, startRender } from '../api/client'
import { LapTable } from '../components/LapTable'
import { MapPreview } from '../components/MapPreview'
import { TelemetryChart } from '../components/TelemetryChart'
import { VideoTrimmer } from '../components/VideoTrimmer'
import { WidgetPicker } from '../components/WidgetPicker'
import type { LapRow, RenderStyle, TelemetryPoint, VideoMeta, WidgetSelection } from '../types'

interface ConfigurePageProps {
  videoId: string
  videoFile: File
  onRenderStarted: (jobId: string, claimToken: string) => void
}

const DEFAULT_WIDGETS: WidgetSelection = { speedo: true, gg: true, minimap: true }
const DEFAULT_STYLE: RenderStyle = { theme: 'cyberpunk', max_expected_speed_kmh: 85, limit_g: 1.5 }

// Speedo max defaults to the video's own recorded top speed rather than a
// fixed value -- a fixed 85 km/h gauge either clips a faster car's needle
// pinned at max or wastes most of the dial on a slow go-kart lap. The
// margin leaves headroom above the peak sample instead of topping out
// exactly at it, and rounding keeps the gauge's tick labels tidy.
const SPEEDO_SAFETY_MARGIN = 1.06
const SPEEDO_ROUNDING_STEP_KMH = 5
const SPEEDO_MIN_KMH = 20

function computeSpeedoMax(points: TelemetryPoint[]): number {
  const topSpeed = points.reduce((max, p) => Math.max(max, p.speed), 0)
  const withMargin = topSpeed * SPEEDO_SAFETY_MARGIN
  return Math.max(SPEEDO_MIN_KMH, Math.ceil(withMargin / SPEEDO_ROUNDING_STEP_KMH) * SPEEDO_ROUNDING_STEP_KMH)
}

export function ConfigurePage({ videoId, videoFile, onRenderStarted }: ConfigurePageProps) {
  const [meta, setMeta] = useState<VideoMeta | null>(null)
  const [points, setPoints] = useState<TelemetryPoint[]>([])
  const [trimStart, setTrimStart] = useState(0)
  const [trimEnd, setTrimEnd] = useState(0)
  const [startMarker, setStartMarker] = useState<{ lat: number; lon: number } | null>(null)
  const [lapStartTime, setLapStartTime] = useState<number | null>(null)
  const [laps, setLaps] = useState<LapRow[]>([])
  const [detectingLaps, setDetectingLaps] = useState(false)
  const [widgets, setWidgets] = useState<WidgetSelection>(DEFAULT_WIDGETS)
  const [style, setStyle] = useState<RenderStyle>(DEFAULT_STYLE)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const videoUrl = useMemo(() => URL.createObjectURL(videoFile), [videoFile])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const [videoMeta, telemetry] = await Promise.all([getVideo(videoId), getTelemetry(videoId)])
      if (cancelled) return
      setMeta(videoMeta)
      setPoints(telemetry)
      setTrimEnd(videoMeta.duration_sec ?? 0)
      setStyle((s) => ({ ...s, max_expected_speed_kmh: computeSpeedoMax(telemetry) }))
    })()
    return () => {
      cancelled = true
    }
  }, [videoId])

  const handlePickStart = (lat: number, lon: number) => {
    setStartMarker({ lat, lon })
    // find the nearest telemetry sample's timestamp to this coordinate
    let best = points[0]
    let bestDist = Infinity
    for (const p of points) {
      const d = (p.lat - lat) ** 2 + (p.lon - lon) ** 2
      if (d < bestDist) {
        bestDist = d
        best = p
      }
    }
    if (best) setLapStartTime(best.time)
  }

  const handleDetectLaps = async () => {
    if (lapStartTime == null) return
    setDetectingLaps(true)
    setError(null)
    try {
      const result = await detectLaps(videoId, lapStartTime)
      setLaps(result.lap_table)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setDetectingLaps(false)
    }
  }

  const handleRender = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const { job_id, claim_token } = await startRender(videoId, trimStart, trimEnd, widgets, style)
      onRenderStarted(job_id, claim_token)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  if (!meta) return <div className="page">Loading telemetry…</div>

  return (
    <div className="page configure-page">
      <h1>{meta.filename}</h1>

      <section>
        <h2>1. Trim the footage</h2>
        <VideoTrimmer videoUrl={videoUrl} duration={meta.duration_sec ?? 0} trimStart={trimStart} trimEnd={trimEnd} onChange={(s, e) => { setTrimStart(s); setTrimEnd(e) }} />
        <TelemetryChart points={points} trimStart={trimStart} trimEnd={trimEnd} laps={laps} lapStartTime={lapStartTime} />
      </section>

      <section>
        <h2>2. Route + lap timing (optional)</h2>
        <div className="configure-page__map-row">
          <MapPreview points={points} startMarker={startMarker} onPickStart={handlePickStart} />
          <div className="lap-detect">
            <p>{lapStartTime != null ? `Start/finish line: t=${lapStartTime.toFixed(1)}s` : 'Click a point on the map to mark the start/finish line.'}</p>
            <button disabled={lapStartTime == null || detectingLaps} onClick={handleDetectLaps}>
              {detectingLaps ? 'Detecting…' : 'Detect laps'}
            </button>
            {laps.length > 0 && <LapTable videoId={videoId} laps={laps} />}
          </div>
        </div>
      </section>

      <section>
        <h2>3. Choose your telemetry widgets</h2>
        <WidgetPicker widgets={widgets} onWidgetsChange={setWidgets} style={style} onStyleChange={setStyle} hasAccel={meta.has_accel} />
      </section>

      {error && <p className="error-text">{error}</p>}

      <button className="primary-button" disabled={submitting} onClick={handleRender}>
        {submitting ? 'Starting render…' : 'Render video'}
      </button>
    </div>
  )
}
