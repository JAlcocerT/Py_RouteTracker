/** Ported from backend/app/render/hud_layers.py's RenderConfig +
 * config_for_resolution. Canvas2D has no matplotlib-style dpi/inches
 * concept, so widthPx/heightPx alone drive scaling -- see hudRenderer.ts's
 * `fontPx`/`linePx` for how the original's pt-sized fonts/strokes (tuned by
 * eye at the 1600x900 "design" canvas) scale to any other resolution. */
export interface RenderConfig {
  enableSpeedo: boolean
  enableLapInfo: boolean
  enableGg: boolean
  enableMinimap: boolean
  maxExpectedSpeedKmh: number
  limitG: number
  theme: 'cyberpunk' | 'dark_background'
  widthPx: number
  heightPx: number
  ggTrailFrames: number
  minimapTrailFrames: number
}

export const DEFAULT_RENDER_CONFIG: RenderConfig = {
  enableSpeedo: true,
  enableLapInfo: true,
  enableGg: true,
  enableMinimap: true,
  maxExpectedSpeedKmh: 85.0,
  limitG: 1.5,
  theme: 'cyberpunk',
  widthPx: 1600,
  heightPx: 900,
  ggTrailFrames: 15,
  minimapTrailFrames: 150,
}

// Field names match the WidgetPicker/ConfigurePage UI's existing settings
// shape (itself mirroring the old JSON render-request body) so those
// components need no changes.
export interface WidgetSelection {
  speedo: boolean
  lapInfo: boolean
  gg: boolean
  minimap: boolean
}

export interface RenderStyle {
  theme: 'cyberpunk' | 'dark_background'
  max_expected_speed_kmh: number
  limit_g: number
}

/** Everything renderConfigFor computes except widthPx/heightPx, which the
 * render pipeline only learns once it's opened the actual video track --
 * see pipeline.ts's RenderVideoOptions. */
export function hudConfigFor(widgets: WidgetSelection, style: RenderStyle): Omit<RenderConfig, 'widthPx' | 'heightPx'> {
  return {
    enableSpeedo: widgets.speedo,
    enableLapInfo: widgets.lapInfo,
    enableGg: widgets.gg,
    enableMinimap: widgets.minimap,
    maxExpectedSpeedKmh: style.max_expected_speed_kmh,
    limitG: style.limit_g,
    theme: style.theme,
    ggTrailFrames: DEFAULT_RENDER_CONFIG.ggTrailFrames,
    minimapTrailFrames: DEFAULT_RENDER_CONFIG.minimapTrailFrames,
  }
}
