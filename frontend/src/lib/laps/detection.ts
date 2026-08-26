/**
 * Lap detection: ported from backend/app/laps/detection.py. See that file's
 * module docstring for the full rationale (start/finish zone + heading-based
 * gate-crossing interpolation, not a naive "closest sample to the start
 * point" trigger).
 */

export interface TelemetryRow {
  time: number
  lat: number
  lon: number
  speed: number
  lat_g: number
  lon_g: number
}

export interface AnnotatedRow extends TelemetryRow {
  lap: number
  last_lap_s: number
  lap_elapsed_s: number
}

export interface LapTableRow {
  lap: number
  start_time: number
  end_time: number
  duration: number
  avg_speed: number
  max_speed: number
}

export interface LapDetectionResult {
  annotatedDf: AnnotatedRow[]
  lapIndices: number[]
  lapTable: LapTableRow[]
}

const EARTH_RADIUS_M = 6_371_000

/** Great-circle distance in meters. Accepts scalars or same-length arrays. */
export function haversineDistanceM(
  lat1: number | number[],
  lon1: number | number[],
  lat2: number,
  lon2: number,
): number | number[] {
  const compute = (la1: number, lo1: number) => {
    const phi1 = (la1 * Math.PI) / 180
    const phi2 = (lat2 * Math.PI) / 180
    const dphi = ((lat2 - la1) * Math.PI) / 180
    const dlambda = ((lon2 - lo1) * Math.PI) / 180
    const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  }
  if (Array.isArray(lat1)) {
    const lons = lon1 as number[]
    return lat1.map((la, i) => compute(la, lons[i]))
  }
  return compute(lat1, lon1 as number)
}

/** Finds the (lat, lon) of the sample closest to `targetTime` seconds. */
export function getCoordinatesAtTime(rows: TelemetryRow[], targetTime: number): [number, number] {
  let bestIdx = 0
  let bestDiff = Infinity
  for (let i = 0; i < rows.length; i++) {
    const diff = Math.abs(rows[i].time - targetTime)
    if (diff < bestDiff) {
      bestDiff = diff
      bestIdx = i
    }
  }
  return [rows[bestIdx].lat, rows[bestIdx].lon]
}

/** Equirectangular (east_m, north_m) of each point relative to an origin --
 * a flat-plane approximation, fine at track scale. */
function localMeters(
  lat: number[],
  lon: number[],
  originLat: number,
  originLon: number,
): { eastM: number[]; northM: number[] } {
  const originLatRad = (originLat * Math.PI) / 180
  const eastM = lon.map((lo) => (((lo - originLon) * Math.PI) / 180) * EARTH_RADIUS_M * Math.cos(originLatRad))
  const northM = lat.map((la) => (((la - originLat) * Math.PI) / 180) * EARTH_RADIUS_M)
  return { eastM, northM }
}

/** Unit (east, north) direction of travel around sample `idx`, from the net
 * displacement over +/- `window` samples. `null` if not actually moving. */
export function _estimateHeading(
  eastM: number[],
  northM: number[],
  idx: number,
  window: number,
): [number, number] | null {
  const lo = Math.max(0, idx - window)
  const hi = Math.min(eastM.length - 1, idx + window)
  const dx = eastM[hi] - eastM[lo]
  const dy = northM[hi] - northM[lo]
  const norm = Math.hypot(dx, dy)
  if (norm < 1e-6) return null
  return [dx / norm, dy / norm]
}

/** Returns [rowIndex, interpolatedTime] of the start/finish crossing. See
 * detection.py's `_resolve_crossing` docstring for the full geometry. */
export function _resolveCrossing(
  eastM: number[],
  northM: number[],
  t: number[],
  zoneStartI: number,
  zoneEndI: number,
  bestIdx: number,
  heading: [number, number] | null,
): [number, number] {
  if (heading !== null) {
    const [hx, hy] = heading
    const lo = Math.max(0, zoneStartI - 1)
    const hi = Math.min(t.length - 1, zoneEndI)
    const proj: number[] = []
    for (let i = lo; i <= hi; i++) proj.push(eastM[i] * hx + northM[i] * hy)
    for (let k = 0; k < proj.length - 1; k++) {
      const p0 = proj[k]
      const p1 = proj[k + 1]
      if (p0 <= 0 !== p1 <= 0) {
        const i0 = lo + k
        const i1 = lo + k + 1
        const frac = p1 === p0 ? 0.0 : Math.min(Math.max(-p0 / (p1 - p0), 0.0), 1.0)
        const crossT = t[i0] + frac * (t[i1] - t[i0])
        return [frac < 0.5 ? i0 : i1, crossT]
      }
    }
  }
  return [bestIdx, t[bestIdx]]
}

export function detectLaps(
  rows: TelemetryRow[],
  startLat: number,
  startLon: number,
  radiusM = 15.0,
  minLapTimeS = 30.0,
): LapDetectionResult {
  if (rows.length === 0) {
    return { annotatedDf: [], lapIndices: [], lapTable: [] }
  }

  const t = rows.map((r) => r.time)
  const lat = rows.map((r) => r.lat)
  const lon = rows.map((r) => r.lon)
  const distToStart = haversineDistanceM(lat, lon, startLat, startLon) as number[]
  const { eastM, northM } = localMeters(lat, lon, startLat, startLon)

  const diffs = t.slice(1).map((v, i) => v - t[i])
  const medianDt = diffs.length ? median(diffs) : 0.0
  const headingWindow = medianDt > 0 ? Math.max(3, Math.round(0.5 / medianDt)) : 3
  let heading: [number, number] | null = null

  const lapIndices: number[] = []
  const crossingTimes: number[] = []
  let lastCrossingTime = -minLapTimeS
  let inZone = false
  let bestDist = Infinity
  let bestIdx = -1
  let zoneStartI = -1

  for (let i = 0; i < rows.length; i++) {
    const rowTime = t[i]
    const dist = distToStart[i]
    if (rowTime - lastCrossingTime > minLapTimeS) {
      if (dist < radiusM) {
        if (!inZone) zoneStartI = i
        inZone = true
        if (dist < bestDist) {
          bestDist = dist
          bestIdx = i
        }
      } else if (inZone) {
        const [crossIdx, crossT] = _resolveCrossing(eastM, northM, t, zoneStartI, i, bestIdx, heading)
        if (heading === null) heading = _estimateHeading(eastM, northM, bestIdx, headingWindow)
        lapIndices.push(crossIdx)
        crossingTimes.push(crossT)
        lastCrossingTime = crossT
        inZone = false
        bestDist = Infinity
      }
    }
  }

  const lapTable: LapTableRow[] = []
  for (let k = 1; k < lapIndices.length; k++) {
    const sIdx = lapIndices[k - 1]
    const eIdx = lapIndices[k]
    const lapSlice = rows.slice(sIdx, eIdx)
    const speeds = lapSlice.map((r) => r.speed)
    lapTable.push({
      lap: k,
      start_time: crossingTimes[k - 1],
      end_time: crossingTimes[k],
      duration: crossingTimes[k] - crossingTimes[k - 1],
      avg_speed: speeds.reduce((a, b) => a + b, 0) / (speeds.length || 1),
      max_speed: speeds.length ? Math.max(...speeds) : NaN,
    })
  }

  const lap = new Array<number>(rows.length).fill(0)
  const lastLapS = new Array<number>(rows.length).fill(0.0)
  let currentLap = 1
  let prevIdx = 0
  for (let k = 0; k < lapIndices.length; k++) {
    const idx = lapIndices[k]
    for (let i = prevIdx; i <= idx; i++) lap[i] = currentLap
    if (prevIdx > 0) {
      const value = crossingTimes[k] - crossingTimes[k - 1]
      for (let i = idx; i < rows.length; i++) lastLapS[i] = value
    }
    prevIdx = idx
    currentLap += 1
  }
  for (let i = prevIdx; i < rows.length; i++) lap[i] = currentLap

  // Live "time since the current lap's own start crossing" for every row --
  // see detection.py's comment on why this is a separate, vectorized pass
  // (most-recent-crossing-at-or-before-this-row's-timestamp) rather than
  // folded into the loop above.
  const lapElapsedS = rows.map((r) => {
    if (crossingTimes.length === 0) return r.time
    let pos = upperBound(crossingTimes, r.time)
    const segmentStart = pos > 0 ? crossingTimes[Math.min(pos - 1, crossingTimes.length - 1)] : 0.0
    return r.time - segmentStart
  })

  const annotatedDf: AnnotatedRow[] = rows.map((r, i) => ({
    ...r,
    lap: lap[i],
    last_lap_s: lastLapS[i],
    lap_elapsed_s: lapElapsedS[i],
  }))

  return { annotatedDf, lapIndices, lapTable }
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  const m = sorted.length
  return m % 2 === 1 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2
}

/** First index where `arr[index] > value` (np.searchsorted(..., side="right")). */
function upperBound(arr: number[], value: number): number {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] <= value) lo = mid + 1
    else hi = mid
  }
  return lo
}
