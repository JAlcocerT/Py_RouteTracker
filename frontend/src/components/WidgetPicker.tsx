import type { RenderStyle, WidgetSelection } from '../types'

interface WidgetPickerProps {
  widgets: WidgetSelection
  onWidgetsChange: (w: WidgetSelection) => void
  style: RenderStyle
  onStyleChange: (s: RenderStyle) => void
  hasAccel: boolean
}

const WIDGET_LABELS: { key: keyof WidgetSelection; label: string; hint?: string }[] = [
  { key: 'speedo', label: 'Speedometer + lap timer' },
  { key: 'gg', label: 'G-G diagram', hint: 'needs accelerometer data' },
  { key: 'minimap', label: 'Minimap' },
]

export function WidgetPicker({ widgets, onWidgetsChange, style, onStyleChange, hasAccel }: WidgetPickerProps) {
  return (
    <div className="widget-picker">
      <div className="widget-picker__widgets">
        {WIDGET_LABELS.map(({ key, label, hint }) => {
          const disabled = key === 'gg' && !hasAccel
          return (
            <label key={key} className={`widget-toggle ${disabled ? 'widget-toggle--disabled' : ''}`}>
              <input
                type="checkbox"
                checked={widgets[key] && !disabled}
                disabled={disabled}
                onChange={(e) => onWidgetsChange({ ...widgets, [key]: e.target.checked })}
              />
              <span>{label}</span>
              {disabled && hint && <small>({hint})</small>}
            </label>
          )
        })}
      </div>

      <div className="widget-picker__style">
        <label className="style-field">
          <span>Theme</span>
          <select value={style.theme} onChange={(e) => onStyleChange({ ...style, theme: e.target.value as RenderStyle['theme'] })}>
            <option value="cyberpunk">Cyberpunk (neon)</option>
            <option value="dark_background">Dark (minimal)</option>
          </select>
        </label>
        <label className="style-field">
          <span>Speedo max (km/h)</span>
          <input
            type="number"
            min={20}
            max={400}
            value={style.max_expected_speed_kmh}
            onChange={(e) => onStyleChange({ ...style, max_expected_speed_kmh: Number(e.target.value) })}
          />
        </label>
        <label className="style-field">
          <span>G-G scale limit</span>
          <input
            type="number"
            min={0.5}
            max={5}
            step={0.1}
            value={style.limit_g}
            onChange={(e) => onStyleChange({ ...style, limit_g: Number(e.target.value) })}
          />
        </label>
      </div>
    </div>
  )
}
