/**
 * Telemetry source for cameras with no embedded GPS track (a standalone GPX
 * file instead). Ported from backend/app/telemetry/sources/external_gpx.py
 * -- see that file's docstring for the alignment rationale.
 *
 * `gpxpy`'s full XML parsing is replaced with a targeted regex scan over
 * `<trkpt>` blocks rather than `DOMParser` -- a real-world GPX track can run
 * to tens of thousands of points, and a full XML DOM parse (jsdom measured
 * well over a minute on this module's own 8.7k-point test fixture) doesn't
 * scale the way this simple, regular, attribute-plus-two-child-elements
 * structure needs to. GPX itself doesn't require anything a DOM would buy
 * here: no nesting below `<trkpt>`, no namespaced siblings to disambiguate.
 */
import { haversineDistanceM } from '../laps/detection'
import { resampleToGrid, smoothSpeedOutliers } from './resample'
import { emptyResult, type TelemetryResult } from './base'

export interface GpxPoint {
  timestamp: Date
  lat: number
  lon: number
  ele: number | null
}

const TRKPT_RE = /<trkpt\b([^>]*)>([\s\S]*?)<\/trkpt>/g
const LAT_ATTR_RE = /\blat="([^"]+)"/
const LON_ATTR_RE = /\blon="([^"]+)"/
const TIME_RE = /<time>([^<]+)<\/time>/
const ELE_RE = /<ele>([^<]+)<\/ele>/

/** Flattens every track/segment point in a GPX file into one array, sorted
 * by timestamp. */
export function loadGpxPoints(gpxText: string): GpxPoint[] {
  const points: GpxPoint[] = []
  for (const match of gpxText.matchAll(TRKPT_RE)) {
    const [, attrs, body] = match
    const latMatch = LAT_ATTR_RE.exec(attrs)
    const lonMatch = LON_ATTR_RE.exec(attrs)
    const timeMatch = TIME_RE.exec(body)
    if (!latMatch || !lonMatch || !timeMatch) continue
    const timestamp = new Date(timeMatch[1])
    if (Number.isNaN(timestamp.getTime())) continue
    const eleMatch = ELE_RE.exec(body)
    points.push({
      timestamp,
      lat: Number(latMatch[1]),
      lon: Number(lonMatch[1]),
      ele: eleMatch ? Number(eleMatch[1]) : null,
    })
  }

  points.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
  return points
}

/** Derives speed from consecutive GPX points (most consumer GPX files don't
 * carry a <speed> extension, so this is computed, not read). Position-delta
 * speed is much noisier than a GPS chip's own reported speed, so it's
 * smoothed the same way as the other sources. */
export function computeSpeedKmh(points: GpxPoint[]): number[] {
  const lat = points.map((p) => p.lat)
  const lon = points.map((p) => p.lon)
  const seconds = points.map((p) => p.timestamp.getTime() / 1000)

  const distM = new Array(points.length).fill(0)
  for (let i = 1; i < points.length; i++) {
    distM[i] = haversineDistanceM(lat[i - 1], lon[i - 1], lat[i], lon[i]) as number
  }

  const dt = seconds.map((s, i) => (i === 0 ? s - (seconds[0] ?? 0) : s - seconds[i - 1]))
  const speedMs = distM.map((d, i) => {
    const denom = dt[i]
    return denom > 0 ? d / denom : 0.0
  })
  const speedKmh = speedMs.map((v) => v * 3.6)
  return smoothSpeedOutliers(speedKmh, 5, 5.0, seconds)
}

export interface ExternalGpxOptions {
  targetFps?: number
  offsetSec?: number
  videoStartTime?: Date
}

export function extractExternalGpx(gpxText: string, durationSec: number, opts: ExternalGpxOptions = {}): TelemetryResult {
  const targetFps = opts.targetFps ?? 30.0
  const offsetSec = opts.offsetSec ?? 0.0

  const points = loadGpxPoints(gpxText)
  if (points.length === 0) return emptyResult('external_gpx')

  const speed = computeSpeedKmh(points)
  const referenceTime = opts.videoStartTime ?? points[0].timestamp

  const withTime = points.map((p, i) => ({
    lat: p.lat,
    lon: p.lon,
    speed: speed[i],
    time: (p.timestamp.getTime() - referenceTime.getTime()) / 1000 + offsetSec,
  }))

  const windowed = withTime.filter((p) => p.time >= 0 && p.time <= durationSec)
  if (windowed.length < 2) return emptyResult('external_gpx')

  const resampled = resampleToGrid(windowed, durationSec, targetFps, ['lat', 'lon', 'speed'])
  const rows = resampled.map((r) => ({
    time: r.time,
    lat: r.lat,
    lon: r.lon,
    speed: r.speed,
    lat_g: 0.0,
    lon_g: 0.0,
  }))
  return { rows, sourceName: 'external_gpx', hasAccel: false }
}
