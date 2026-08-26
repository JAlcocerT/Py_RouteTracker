/**
 * GoPro embedded-metadata telemetry source. Ported from
 * backend/app/telemetry/sources/gopro_embedded.py -- but the extraction
 * mechanism itself necessarily changes: there is no browser-side exiftool,
 * so instead of dumping+regex-parsing an `exiftool -ee` text report, this
 * reads the raw GPMF track directly via `gpmf-extract` (mp4box-based,
 * built for exactly this browser use case) and decodes it with
 * `gopro-telemetry` (the same author's GPMF parser, also browser-tested).
 *
 * This is a genuine algorithm swap, not a line-for-line port -- flagged and
 * accepted when this rewrite was planned. It hasn't been validated against
 * a real GoPro file end-to-end (no such fixture exists in this repo); the
 * GPS5/GPS9 sample layout and the ACCL axis order below follow gopro-
 * telemetry's documented output shape and GPMF's own documented axis
 * convention, but this module needs a manual test pass against real
 * footage before being trusted the way the exiftool path was.
 */
import gpmfExtract from 'gpmf-extract'
import goproTelemetry from 'gopro-telemetry'
import { emptyResult, type TelemetryResult } from './base'
import { resampleToGrid, rollingMeanCentered, smoothSpeedOutliers } from './resample'

const ACCEL_SMOOTHING_WINDOW = 15

// gopro-telemetry's own .d.ts models its result as a conditional type keyed
// off the exact options object passed in, via internal interfaces (
// GoProTelemetryPresetLessOptions, GoProTelemetryStreamResult, ...) that
// aren't exported from the package -- there's no way to name them to thread
// real inference through. This is a plain, manually-written mirror of the
// GPS5/GPS9/ACCL shapes documented in that .d.ts (units 'deg'/'deg'/'m'/
// 'm/s'/'m/s' for GPS5/GPS9's [lat, lon, alt, speed2D, speed3D], 'm/s²' for
// ACCL's [z, x, y]) and cast to it once, at the library boundary below.
interface GpsSample {
  cts: number // milliseconds since the start of the video
  value: [number, number, number, number, number]
}
interface AccelSample {
  cts: number
  value: [number, number, number]
}
interface GoproDeviceData {
  streams: {
    GPS5?: { samples: GpsSample[] }
    GPS9?: { samples: GpsSample[] }
    ACCL?: { samples: AccelSample[] }
  }
}
type GoproTelemetryResult = Record<number, GoproDeviceData>

function findGpsSamples(telemetry: GoproTelemetryResult): GpsSample[] {
  for (const device of Object.values(telemetry)) {
    const stream = device.streams.GPS9 ?? device.streams.GPS5
    if (stream?.samples?.length) return stream.samples
  }
  return []
}

function findAccelSamples(telemetry: GoproTelemetryResult): AccelSample[] {
  for (const device of Object.values(telemetry)) {
    if (device.streams.ACCL?.samples?.length) return device.streams.ACCL.samples
  }
  return []
}

/** GPS5 = [lat, lon, altitude, speed2D, speed3D]; GPS9 carries the same
 * first 5 fields plus fix-quality metadata this app doesn't use. Ground
 * speed (not 3D, which includes vertical rate) is what a "GPS Speed"
 * readout conventionally means. */
function gpsRowsFromSamples(samples: GpsSample[]): { time: number; lat: number; lon: number; speed: number }[] {
  const rows = samples
    .map((s) => ({
      time: s.cts / 1000,
      lat: s.value[0],
      lon: s.value[1],
      speed: s.value[3] * 3.6, // m/s -> km/h
    }))
    // (0, 0) is GPMF's "no fix yet" sentinel, same as the exiftool path.
    .filter((r) => r.lat !== 0 || r.lon !== 0)
  if (rows.length < 2) return []

  const time = rows.map((r) => r.time)
  const smoothedSpeed = smoothSpeedOutliers(
    rows.map((r) => r.speed),
    5,
    5.0,
    time,
  )
  return rows.map((r, i) => ({ ...r, speed: smoothedSpeed[i] }))
}

/** GPMF's documented ACCL axis order is [Z, X, Y] -- Z (vertical, gravity-
 * dominant) is skipped, X/Y feed lat_g/lon_g, matching which raw components
 * the original ffmpeg-dumped-binary parser used (see gopro_embedded.py's
 * `parse_gpmd_accel`). Un-calibrated: each component is normalized by the
 * session's own median acceleration magnitude rather than a fixed physical
 * g constant, exactly like the original -- this only needs to be
 * "approximately 1 at rest / cruising", not absolute units, since it's a
 * ratio. */
function accelRowsFromSamples(samples: AccelSample[]): { time: number; lat_g: number; lon_g: number }[] {
  if (samples.length === 0) return []

  const magnitudes = samples.map((s) => Math.hypot(s.value[0], s.value[1], s.value[2]))
  const oneG = median(magnitudes)
  if (!oneG) return []

  const rawLat = samples.map((s) => s.value[1] / oneG)
  const rawLon = samples.map((s) => s.value[2] / oneG)
  const latG = rollingMeanCentered(rawLat, ACCEL_SMOOTHING_WINDOW).map((v) => (Number.isNaN(v) ? 0 : v))
  const lonG = rollingMeanCentered(rawLon, ACCEL_SMOOTHING_WINDOW).map((v) => (Number.isNaN(v) ? 0 : v))

  return samples.map((s, i) => ({ time: s.cts / 1000, lat_g: latG[i], lon_g: lonG[i] }))
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  const m = sorted.length
  if (m === 0) return 0
  return m % 2 === 1 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2
}

// gpmf-extract's default worker-based file reader has a bug: when the video
// has no 'gpmd' track, it kills the reader via the browser's native
// Worker.terminate() (see its index.js's onReady branch) -- which just ends
// the thread silently, firing neither onmessage nor onerror. The extraction
// promise then never settles, so this app's "Extracting telemetry" step
// hangs forever with no error. The non-worker reader aborts through an
// AbortController instead, which does reject correctly, so passing
// useWorker: false both fixes that hang and sidesteps the library's own
// documented "seems to crash on some recent browsers" risk for the worker
// path -- see its index.d.ts. mp4box parsing runs on the main thread either
// way (both readers hand parsed buffers to `mp4boxFile.appendBuffer` there),
// so this isn't trading away any real offload.
const EXTRACTION_TIMEOUT_MS = 5 * 60 * 1000

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms)
    promise.then(
      (v) => {
        clearTimeout(timer)
        resolve(v)
      },
      (e) => {
        clearTimeout(timer)
        reject(e)
      },
    )
  })
}

export interface GoProExtractOptions {
  targetFps?: number
  /** Fraction (0-1) of overall extraction progress: gpmf-extract's file read
   * is weighted 70%, gopro-telemetry's decode the remaining 30% -- both
   * libraries report progress on their own incomparable scales, so this is
   * a rough split, not a measured one. */
  onProgress?: (fraction: number) => void
}

export async function extractGoProGpmf(
  videoFile: File,
  durationSec: number,
  { targetFps = 30.0, onProgress }: GoProExtractOptions = {},
): Promise<TelemetryResult> {
  let extracted: Parameters<typeof goproTelemetry>[0]
  try {
    extracted = await withTimeout(
      gpmfExtract(videoFile, {
        browserMode: true,
        useWorker: false,
        progress: (p) => onProgress?.((p / 100) * 0.7),
      }),
      EXTRACTION_TIMEOUT_MS,
      'Timed out reading the video file -- it may be corrupted or unusually large.',
    )
  } catch (e) {
    // Plain string rejections ('Track not found' / 'File not compatible')
    // are gpmf-extract's own no-gpmd-track signal, not a real failure --
    // treat them the same as "found the track but no usable GPS samples".
    if (e === 'Track not found' || e === 'File not compatible') return emptyResult('gopro_embedded')
    throw e
  }

  const result = await withTimeout(
    goproTelemetry(extracted, { stream: ['GPS', 'ACCL'], progress: (p) => onProgress?.(0.7 + p * 0.3) }),
    EXTRACTION_TIMEOUT_MS,
    'Timed out decoding GPS telemetry from the video.',
  )
  const telemetry = result as unknown as GoproTelemetryResult

  const gpsRows = gpsRowsFromSamples(findGpsSamples(telemetry))
  if (gpsRows.length === 0) return emptyResult('gopro_embedded')

  const accelRows = accelRowsFromSamples(findAccelSamples(telemetry))
  const hasAccel = accelRows.length > 0

  const gpsResampled = resampleToGrid(gpsRows, durationSec, targetFps, ['lat', 'lon', 'speed'])
  const accelResampled = hasAccel ? resampleToGrid(accelRows, durationSec, targetFps, ['lat_g', 'lon_g']) : null

  const rows = gpsResampled.map((r, i) => ({
    time: r.time,
    lat: r.lat,
    lon: r.lon,
    speed: r.speed,
    lat_g: accelResampled ? accelResampled[i].lat_g : 0.0,
    lon_g: accelResampled ? accelResampled[i].lon_g : 0.0,
  }))

  onProgress?.(1)
  return { rows, sourceName: 'gopro_embedded', hasAccel }
}
