import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { LapRow, TelemetryPoint } from '../types'

interface TelemetryChartProps {
  points: TelemetryPoint[]
  trimStart?: number
  trimEnd?: number
  laps?: LapRow[]
  lapStartTime?: number | null
}

export function TelemetryChart({ points, trimStart, trimEnd, laps, lapStartTime }: TelemetryChartProps) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={points} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
        <XAxis dataKey="time" tickFormatter={(t: number) => `${Math.round(t)}s`} stroke="#888" fontSize={11} />
        <YAxis stroke="#888" fontSize={11} unit=" km/h" width={60} />
        <Tooltip
          contentStyle={{ background: '#161622', border: '1px solid #333', fontSize: 12 }}
          labelFormatter={(t) => `t=${Number(t).toFixed(1)}s`}
        />
        {trimStart !== undefined && trimEnd !== undefined && (
          <ReferenceArea x1={trimStart} x2={trimEnd} fill="#00ff9f" fillOpacity={0.08} />
        )}
        {lapStartTime != null && <ReferenceLine x={lapStartTime} stroke="#ffe14d" strokeDasharray="4 4" label={{ value: 'start', fill: '#ffe14d', fontSize: 10 }} />}
        {laps?.map((lap) => (
          <ReferenceLine key={lap.lap} x={lap.end_time} stroke="#4dd2ff" strokeOpacity={0.4} />
        ))}
        <Line type="monotone" dataKey="speed" stroke="#00ff9f" dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
