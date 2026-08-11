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
  onProgress?: (fraction: number) => void
}

export async function uploadVideo(opts: UploadOptions): Promise<{ video_id: string; job_id: string; duration_sec: number }> {
  const form = new FormData()
  form.append('video', opts.video)
  form.append('source_type', opts.sourceType)
  if (opts.gpx) form.append('gpx', opts.gpx)
  if (opts.videoStartTime) form.append('video_start_time', opts.videoStartTime)
  if (opts.offsetSec !== undefined) form.append('offset_sec', String(opts.offsetSec))

  // fetch() has no upload-progress event, only XHR does (xhr.upload.onprogress) --
  // needed here since video files can be large enough that "uploading" is a
  // real, multi-second phase the user shouldn't stare at a blank button for.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/videos')

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress) opts.onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          reject(new Error('Upload succeeded but the response was not valid JSON'))
        }
      } else {
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${xhr.responseText}`))
      }
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))

    xhr.send(form)
  })
}

export async function getVideo(videoId: string): Promise<VideoMeta> {
  return asJson(await fetch(`/api/videos/${videoId}`))
}

export async function getHealth(): Promise<{ status: string; retention_minutes: number }> {
  return asJson(await fetch('/api/health'))
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
): Promise<{ job_id: string; claim_token: string }> {
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
