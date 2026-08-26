/**
 * Shared time-grid resampling, used by every telemetry source. Ported from
 * app/telemetry/resample.py -- see that file's docstrings for the reasoning
 * behind the Hampel filter and the per-source-rate window sizing.
 */

/** pandas' `.rolling(window, center=True, min_periods=1)` window bounds:
 * for output index i, the window spans [i - leftOffset, i + rightOffset]
 * clipped to the array, where rightOffset = floor((window-1)/2) and
 * leftOffset = (window-1) - rightOffset. Verified empirically against
 * pandas for both odd and even window sizes (both occur here -- window is
 * derived from a sample-rate ratio and rounded, so it can land on either).
 */
function rollingMedianCentered(values: number[], window: number): number[] {
  const n = values.length
  const rightOffset = Math.floor((window - 1) / 2)
  const leftOffset = window - 1 - rightOffset
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - leftOffset)
    const hi = Math.min(n - 1, i + rightOffset)
    const window_ = values.slice(lo, hi + 1).slice().sort((a, b) => a - b)
    const m = window_.length
    out[i] = m % 2 === 1 ? window_[(m - 1) / 2] : (window_[m / 2 - 1] + window_[m / 2]) / 2
  }
  return out
}

/** pandas' `.rolling(window, center=True).mean()` (no `min_periods`
 * override, so it defaults to requiring the *full* window) -- unlike
 * `rollingMedianCentered` above, an index whose clipped window would be
 * short is NaN, not a partial-window average. Used by the GoPro
 * accelerometer smoothing, which fills those edge NaNs with 0 afterwards. */
export function rollingMeanCentered(values: number[], window: number): number[] {
  const n = values.length
  const rightOffset = Math.floor((window - 1) / 2)
  const leftOffset = window - 1 - rightOffset
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const lo = i - leftOffset
    const hi = i + rightOffset
    if (lo < 0 || hi > n - 1) {
      out[i] = NaN
      continue
    }
    let sum = 0
    for (let j = lo; j <= hi; j++) sum += values[j]
    out[i] = sum / window
  }
  return out
}

/** Rejects implausible GPS speed spikes with a Hampel filter (rolling
 * median + median-absolute-deviation threshold). See resample.py's
 * `smooth_speed_outliers` docstring for the full rationale. */
export function smoothSpeedOutliers(
  speed: number[],
  window = 5,
  madThreshold = 5.0,
  time?: number[],
  targetWindowSec = 0.5,
): number[] {
  let w = window
  if (time && time.length >= 2) {
    const diffs = time.slice(1).map((t, i) => t - time[i])
    const medianDt = median(diffs)
    if (medianDt > 0) w = Math.max(3, Math.round(targetWindowSec / medianDt))
  }
  if (speed.length < w) return speed.slice()

  const localMedian = rollingMedianCentered(speed, w)
  const deviation = speed.map((v, i) => Math.abs(v - localMedian[i]))
  const mad = rollingMedianCentered(deviation, w)
  // 1.4826 makes MAD comparable to a standard deviation for normally
  // distributed data; floored so a near-constant local window (MAD ~ 0)
  // doesn't make ordinary tiny wobble register as an outlier.
  const scaledMad = mad.map((m) => Math.max(m * 1.4826, 1.0))
  return speed.map((v, i) => (deviation[i] > madThreshold * scaledMad[i] ? localMedian[i] : v))
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  const m = sorted.length
  if (m === 0) return NaN
  return m % 2 === 1 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2
}

/** Interpolates `valueCols` (indexed by each row's `time`) onto a uniform
 * grid from 0 to durationSec in steps of 1/targetFps.
 *
 * Equivalent to the Python version's
 * `reindex(union).interpolate(method="index").ffill().bfill().reindex(target)`:
 * for each target time, linearly interpolate (by real time, not position)
 * between the two nearest known samples, clamping to the first/last known
 * value outside the samples' own time range.
 */
export function resampleToGrid<T extends Record<string, number>>(
  rows: (T & { time: number })[],
  durationSec: number,
  targetFps: number,
  valueCols: (keyof T & string)[],
): (Record<string, number> & { time: number })[] {
  const targetTimes: number[] = []
  for (let t = 0; t < durationSec; t += 1 / targetFps) targetTimes.push(t)

  if (rows.length === 0) {
    return targetTimes.map((time) => {
      const row: Record<string, number> = { time }
      for (const col of valueCols) row[col] = NaN
      return row as Record<string, number> & { time: number }
    })
  }

  const sorted = rows.slice().sort((a, b) => a.time - b.time)
  const times = sorted.map((r) => r.time)

  return targetTimes.map((t) => {
    const row: Record<string, number> = { time: t }
    // First index with times[idx] >= t (lower_bound).
    let lo = 0
    let hi = times.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (times[mid] < t) lo = mid + 1
      else hi = mid
    }
    for (const col of valueCols) {
      if (lo <= 0) {
        row[col] = sorted[0][col]
      } else if (lo >= times.length) {
        row[col] = sorted[times.length - 1][col]
      } else if (times[lo] === t) {
        row[col] = sorted[lo][col]
      } else {
        const t0 = times[lo - 1]
        const t1 = times[lo]
        const v0 = sorted[lo - 1][col]
        const v1 = sorted[lo][col]
        const frac = (t - t0) / (t1 - t0)
        row[col] = v0 + frac * (v1 - v0)
      }
    }
    return row as Record<string, number> & { time: number }
  })
}
