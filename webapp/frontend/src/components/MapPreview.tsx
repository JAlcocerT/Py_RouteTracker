import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import { useMemo } from 'react'
import { MapContainer, Marker, Polyline, TileLayer, useMapEvents } from 'react-leaflet'
import type { TelemetryPoint } from '../types'

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
})

interface MapPreviewProps {
  points: TelemetryPoint[]
  startMarker: { lat: number; lon: number } | null
  onPickStart: (lat: number, lon: number) => void
}

function ClickHandler({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onPick(e.latlng.lat, e.latlng.lng)
    },
  })
  return null
}

export function MapPreview({ points, startMarker, onPickStart }: MapPreviewProps) {
  const path = useMemo<[number, number][]>(() => points.map((p) => [p.lat, p.lon]), [points])
  const center = useMemo<[number, number]>(() => {
    if (path.length === 0) return [0, 0]
    return path[Math.floor(path.length / 2)]
  }, [path])

  if (points.length === 0) {
    return <div className="map-preview map-preview--empty">No telemetry yet</div>
  }

  return (
    <div className="map-preview">
      <MapContainer center={center} zoom={15} className="map-preview__map">
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Polyline positions={path} pathOptions={{ color: '#00ff9f', weight: 3 }} />
        <ClickHandler onPick={onPickStart} />
        {startMarker && <Marker position={[startMarker.lat, startMarker.lon]} />}
      </MapContainer>
      <p className="map-preview__hint">Click the map to place the lap start/finish line.</p>
    </div>
  )
}
