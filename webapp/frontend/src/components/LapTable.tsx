import { useState } from 'react'
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { compareLaps } from '../api/client'
import type { LapComparison, LapRow } from '../types'

interface LapTableProps {
  videoId: string
  laps: LapRow[]
}

function fmt(sec: number): string {
  return `${sec.toFixed(2)}s`
}

export function LapTable({ videoId, laps }: LapTableProps) {
  const [lapA, setLapA] = useState<number | null>(laps[0]?.lap ?? null)
  const [lapB, setLapB] = useState<number | null>(laps[1]?.lap ?? null)
  const [comparison, setComparison] = useState<LapComparison | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const bestLap = laps.reduce<LapRow | null>((best, l) => (!best || l.duration < best.duration ? l : best), null)

  const handleCompare = async () => {
    if (lapA == null || lapB == null || lapA === lapB) return
    setLoading(true)
    setError(null)
    try {
      setComparison(await compareLaps(videoId, lapA, lapB))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="lap-table">
      <table>
        <thead>
          <tr>
            <th>Lap</th>
            <th>Duration</th>
            <th>Avg speed</th>
            <th>Max speed</th>
          </tr>
        </thead>
        <tbody>
          {laps.map((lap) => (
            <tr key={lap.lap} className={bestLap?.lap === lap.lap ? 'lap-table__best' : ''}>
              <td>{lap.lap} {bestLap?.lap === lap.lap && '🔥'}</td>
              <td>{fmt(lap.duration)}</td>
              <td>{lap.avg_speed.toFixed(1)} km/h</td>
              <td>{lap.max_speed.toFixed(1)} km/h</td>
            </tr>
          ))}
        </tbody>
      </table>

      {laps.length >= 2 && (
        <div className="lap-compare">
          <div className="lap-compare__controls">
            <select value={lapA ?? ''} onChange={(e) => setLapA(Number(e.target.value))}>
              {laps.map((l) => (
                <option key={l.lap} value={l.lap}>Lap {l.lap}</option>
              ))}
            </select>
            <span>vs</span>
            <select value={lapB ?? ''} onChange={(e) => setLapB(Number(e.target.value))}>
              {laps.map((l) => (
                <option key={l.lap} value={l.lap}>Lap {l.lap}</option>
              ))}
            </select>
            <button onClick={handleCompare} disabled={loading || lapA === lapB}>
              {loading ? 'Comparing…' : 'Compare'}
            </button>
          </div>
          {error && <p className="error-text">{error}</p>}

          {comparison && (
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis dataKey="rel_time" type="number" tickFormatter={(t: number) => `${t.toFixed(0)}s`} stroke="#888" fontSize={11} />
                <YAxis dataKey="speed" stroke="#888" fontSize={11} unit=" km/h" width={60} />
                <Tooltip contentStyle={{ background: '#161622', border: '1px solid #333', fontSize: 12 }} />
                <Line data={comparison.series_a} dataKey="speed" name={`Lap ${comparison.lap_a} (${fmt(comparison.duration_a)})`} stroke="#4dd2ff" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line data={comparison.series_b} dataKey="speed" name={`Lap ${comparison.lap_b} (${fmt(comparison.duration_b)})`} stroke="#ff0055" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Scatter data={comparison.maxima_a.map(([t, s]) => ({ rel_time: t, speed: s }))} fill="#4dd2ff" shape="triangle" />
                <Scatter data={comparison.minima_a.map(([t, s]) => ({ rel_time: t, speed: s }))} fill="#4dd2ff" shape="diamond" />
                <Scatter data={comparison.maxima_b.map(([t, s]) => ({ rel_time: t, speed: s }))} fill="#ff0055" shape="triangle" />
                <Scatter data={comparison.minima_b.map(([t, s]) => ({ rel_time: t, speed: s }))} fill="#ff0055" shape="diamond" />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  )
}
