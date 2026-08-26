/**
 * Common contract for telemetry sources. Ported from
 * backend/app/telemetry/sources/base.py.
 */
import type { TelemetryRow } from '../laps/detection'

export interface TelemetryResult {
  rows: TelemetryRow[]
  sourceName: string
  hasAccel: boolean
}

export function emptyResult(sourceName: string): TelemetryResult {
  return { rows: [], sourceName, hasAccel: false }
}
