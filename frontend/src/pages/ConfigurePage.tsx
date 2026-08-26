import { useEffect, useMemo, useState } from 'react'
import { LapTable } from '../components/LapTable'
import { MapPreview } from '../components/MapPreview'
import { TelemetryChart } from '../components/TelemetryChart'
import { VideoTrimmer } from '../components/VideoTrimmer'
import { WidgetPicker } from '../components/WidgetPicker'
import { detectLaps, type AnnotatedRow } from '../lib/laps/detection'
import type { LapRow, RenderStyle, TelemetryPoint, WidgetSelection } from '../types'

interface ConfigurePageProps {
  videoFile: File
  duration: number
  telemetry: TelemetryPoint[]
  hasAccel: boolean
  onRenderStarted: (payload: {
    videoFile: File
    trimStart: number
    trimEnd: number
    widgets: WidgetSelection
    style: RenderStyle
    annotatedRows: AnnotatedRow[]
  }) => void
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

/** Matches app.render.coordinator._prepare_claimed_job's fallback for a
 * video that never had lap detection run on it: flat zeros, not an error --
 * lap widgets are opt-in, not a hard requirement to render at all. */
function withoutLapAnnotation(points: TelemetryPoint[]): AnnotatedRow[] {
  return points.map((p) => ({ ...p, lap: 0, last_lap_s: 0, lap_elapsed_s: 0 }))
}

export function ConfigurePage({ videoFile, duration, telemetry, hasAccel, onRenderStarted }: ConfigurePageProps) {
  const [trimStart, setTrimStart] = useState(0)
  const [trimEnd, setTrimEnd] = useState(duration)
  const [startMarker, setStartMarker] = useState<{ lat: number; lon: number } | null>(null)
  const [lapStartTime, setLapStartTime] = useState<number | null>(null)
  const [laps, setLaps] = useState<LapRow[]>([])
  const [lapIndices, setLapIndices] = useState<number[]>([])
  const [annotatedRows, setAnnotatedRows] = useState<AnnotatedRow[]>(() => withoutLapAnnotation(telemetry))
  const [detectingLaps, setDetectingLaps] = useState(false)
  const [widgets, setWidgets] = useState<WidgetSelection>(DEFAULT_WIDGETS)
  const [style, setStyle] = useState<RenderStyle>({ ...DEFAULT_STYLE, max_expected_speed_kmh: computeSpeedoMax(telemetry) })
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const videoUrl = useMemo(() => URL.createObjectURL(videoFile), [videoFile])
  useEffect(() => () => URL.revokeObjectURL(videoUrl), [videoUrl])

  const handlePickStart = (lat: number, lon: number) => {
    setStartMarker({ lat, lon })
    // find the nearest telemetry sample's timestamp to this coordinate
    let best = telemetry[0]
    let bestDist = Infinity
    for (const p of telemetry) {
      const d = (p.lat - lat) ** 2 + (p.lon - lon) ** 2
      if (d < bestDist) {
        bestDist = d
        best = p
      }
    }
    if (best) setLapStartTime(best.time)
  }

  const handleDetectLaps = () => {
    if (lapStartTime == null) return
    setDetectingLaps(true)
    setError(null)
    try {
      const result = detectLaps(telemetry, startMarker!.lat, startMarker!.lon)
      setLaps(result.lapTable)
      setLapIndices(result.lapIndices)
      setAnnotatedRows(result.annotatedDf)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setDetectingLaps(false)
    }
  }

  const handleRender = () => {
    setSubmitting(true)
    setError(null)
    try {
      onRenderStarted({ videoFile, trimStart, trimEnd, widgets, style, annotatedRows })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="page configure-page">
      <h1>{videoFile.name}</h1>

      <section>
        <h2>1. Trim the footage</h2>
        <VideoTrimmer videoUrl={videoUrl} duration={duration} trimStart={trimStart} trimEnd={trimEnd} onChange={(s, e) => { setTrimStart(s); setTrimEnd(e) }} />
        <TelemetryChart points={telemetry} trimStart={trimStart} trimEnd={trimEnd} laps={laps} lapStartTime={lapStartTime} />
      </section>

      <section>
        <h2>2. Route + lap timing (optional)</h2>
        <div className="configure-page__map-row">
          <MapPreview points={telemetry} startMarker={startMarker} onPickStart={handlePickStart} />
          <div className="lap-detect">
            <p>{lapStartTime != null ? `Start/finish line: t=${lapStartTime.toFixed(1)}s` : 'Click a point on the map to mark the start/finish line.'}</p>
            <button disabled={lapStartTime == null || detectingLaps} onClick={handleDetectLaps}>
              {detectingLaps ? 'Detecting…' : 'Detect laps'}
            </button>
            {laps.length > 0 && <LapTable telemetry={annotatedRows} lapIndices={lapIndices} laps={laps} />}
          </div>
        </div>
      </section>

      <section>
        <h2>3. Choose your telemetry widgets</h2>
        <WidgetPicker widgets={widgets} onWidgetsChange={setWidgets} style={style} onStyleChange={setStyle} hasAccel={hasAccel} />
      </section>

      {error && <p className="error-text">{error}</p>}

      <button className="primary-button" disabled={submitting} onClick={handleRender}>
        {submitting ? 'Starting render…' : 'Render video'}
      </button>
    </div>
  )
}
