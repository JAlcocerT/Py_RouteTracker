import type {
  JobStatus,
  LapComparison,
  LapRow,
  RenderStyle,
  SourceType,
  TelemetryPoint,
  VideoMeta,
  WidgetSelection,
} from '../types'

async function asJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const body = await resp.text()
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`)
  }
  return resp.json() as Promise<T>
}

export interface UploadOptions {
  video: File
  sourceType: SourceType
  gpx?: File
  videoStartTime?: string
  offsetSec?: number
}

export async function uploadVideo(opts: UploadOptions): Promise<{ video_id: string; job_id: string; duration_sec: number }> {
  const form = new FormData()
  form.append('video', opts.video)
  form.append('source_type', opts.sourceType)
  if (opts.gpx) form.append('gpx', opts.gpx)
  if (opts.videoStartTime) form.append('video_start_time', opts.videoStartTime)
  if (opts.offsetSec !== undefined) form.append('offset_sec', String(opts.offsetSec))

  const resp = await fetch('/api/videos', { method: 'POST', body: form })
  return asJson(resp)
}

export async function getVideo(videoId: string): Promise<VideoMeta> {
  return asJson(await fetch(`/api/videos/${videoId}`))
}

export async function getTelemetry(videoId: string, maxPoints = 2000): Promise<TelemetryPoint[]> {
  const resp = await fetch(`/api/videos/${videoId}/telemetry?max_points=${maxPoints}`)
  const body = await asJson<{ points: TelemetryPoint[] }>(resp)
  return body.points
}

export async function getJob(jobId: string): Promise<JobStatus> {
  return asJson(await fetch(`/api/jobs/${jobId}`))
}

export async function detectLaps(
  videoId: string,
  startTimeS: number,
  radiusM = 15.0,
  minLapTimeS = 30.0,
): Promise<{ lap_count: number; lap_table: LapRow[]; start_lat: number; start_lon: number }> {
  const resp = await fetch(`/api/videos/${videoId}/laps/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start_time_s: startTimeS, radius_m: radiusM, min_lap_time_s: minLapTimeS }),
  })
  return asJson(resp)
}

export async function compareLaps(videoId: string, lapA: number, lapB: number): Promise<LapComparison> {
  const resp = await fetch(`/api/videos/${videoId}/laps/compare?lap_a=${lapA}&lap_b=${lapB}`)
  return asJson(resp)
}

export async function startRender(
  videoId: string,
  trimStart: number,
  trimEnd: number,
  widgets: WidgetSelection,
  style: RenderStyle,
): Promise<{ job_id: string }> {
  const resp = await fetch(`/api/videos/${videoId}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trim_start: trimStart, trim_end: trimEnd, widgets, style }),
  })
  return asJson(resp)
}

export function downloadUrl(jobId: string): string {
  return `/api/render/${jobId}/download`
}
