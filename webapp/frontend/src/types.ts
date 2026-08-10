export type SourceType = 'gopro_embedded' | 'external_gpx'

export interface VideoMeta {
  id: string
  filename: string
  video_path: string
  source_type: SourceType
  gpx_path: string | null
  duration_sec: number | null
  has_accel: boolean
  telemetry_ready: boolean
  laps_ready: boolean
  extraction_job_id: string | null
  lap_start_time_s: number | null
}

export interface TelemetryPoint {
  time: number
  lat: number
  lon: number
  speed: number
  lat_g: number
  lon_g: number
}

export interface LapRow {
  lap: number
  start_time: number
  end_time: number
  duration: number
  avg_speed: number
  max_speed: number
}

export interface JobStatus {
  id: string
  kind: string
  status: 'pending' | 'running' | 'done' | 'error'
  progress: number
  error: string | null
  result: Record<string, unknown> | null
}

export interface WidgetSelection {
  speedo: boolean
  gg: boolean
  minimap: boolean
  session_graph: boolean
}

export interface RenderStyle {
  theme: 'cyberpunk' | 'dark_background'
  max_expected_speed_kmh: number
  limit_g: number
}

export interface LapComparison {
  lap_a: number
  lap_b: number
  duration_a: number
  duration_b: number
  series_a: { rel_time: number; speed: number }[]
  series_b: { rel_time: number; speed: number }[]
  maxima_a: [number, number][]
  minima_a: [number, number][]
  maxima_b: [number, number][]
  minima_b: [number, number][]
}
