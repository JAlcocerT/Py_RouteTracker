/**
 * Canvas2D port of backend/app/render/hud_layers.py's HudRenderer.
 *
 * matplotlib's blitting optimization (cache the static background, redraw
 * only the handful of artists that move each frame) existed purely to make
 * a 14+ minute render usable -- see that file's module docstring. Canvas2D
 * redrawing everything from scratch every frame is already fast enough
 * that none of that machinery is needed here; `drawFrame` just draws the
 * whole frame each time.
 *
 * matplotlib's data-space math (angles, arc points, the 0..1 gauge-space
 * coordinates) is kept as-is and only converted to canvas pixels at the
 * last step, via `Transform.toPx` -- this is deliberate: it means every
 * formula below can be checked directly against hud_layers.py's own
 * formula for the same artist, rather than re-derived in pixel space.
 *
 * Visual approximations, not pixel-identical to the matplotlib version:
 *  - mplcyberpunk's glow effect is approximated with a canvas `shadowBlur`,
 *    not its actual multi-pass line-alpha algorithm.
 *  - matplotlib's `withStroke` path effect (a black outline behind text/
 *    lines) is approximated by stroking in black before filling.
 * This hasn't been visually compared side-by-side against a real rendered
 * frame from the Python version (no headless-matplotlib environment was
 * available to generate one here) -- worth a manual look once this runs
 * end-to-end in a browser.
 */
import type { AnnotatedRow } from '../laps/detection'
import type { RenderConfig } from './renderConfig'

const OUTLINE_COLOR = '#000000'
const GREEN = '#00ff9f'
const YELLOW = '#ffd400'
const RED = '#ff0055'
const CYAN = '#00e5ff'

const SWEEP_START_DEG = 210.0
const SWEEP_END_DEG = -30.0
const LAST_LAP_PROMPT_SECONDS = 5.0

const PANEL_FACE = 'rgba(5, 13, 15, 0.42)'
const PANEL_EDGE = 'rgba(255, 255, 255, 0.18)'

const SPEEDO_PANEL_RECT = [0.015, 0.64, 0.18, 0.32] as const
const GG_PANEL_RECT = [0.015, 0.035, 0.17, 0.21] as const
const MINIMAP_PANEL_RECT = [0.745, 0.035, 0.24, 0.32] as const
const SPEEDO_RECT = [0.018, 0.656, 0.174, 0.288] as const
const GG_RECT = [0.03, 0.05, 0.15, 0.19] as const
const MINIMAP_RECT = [0.757, 0.047, 0.224, 0.288] as const

const DESIGN_WIDTH_PX = 1600

function angleForFrac(frac: number): number {
  return ((SWEEP_START_DEG + frac * (SWEEP_END_DEG - SWEEP_START_DEG)) * Math.PI) / 180
}

export function formatLapTime(seconds: number): string {
  const totalMs = Math.round(seconds * 1000)
  const minutes = Math.floor(totalMs / 60_000)
  const remMs = totalMs % 60_000
  const secs = Math.floor(remMs / 1000)
  const ms = remMs % 1000
  return `${minutes}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`
}

interface Transform {
  toPx(x: number, y: number): [number, number]
  scale: number
}

/** Figure-fraction (left, bottom, width, height), bottom-left origin like
 * matplotlib, to a canvas pixel box (top-left origin). */
function figRectToPixelBox(rect: readonly [number, number, number, number], widthPx: number, heightPx: number) {
  const [left, bottom, width, height] = rect
  return { x: left * widthPx, y: (1 - bottom - height) * heightPx, w: width * widthPx, h: height * heightPx }
}

/** matplotlib's `set_aspect("equal")` within a (possibly non-square) axes
 * box: the data unit is the same in x and y, and the data is centered in
 * whichever dimension has slack. */
function equalAspectTransform(
  pixelBox: { x: number; y: number; w: number; h: number },
  dataMinX: number,
  dataMaxX: number,
  dataMinY: number,
  dataMaxY: number,
): Transform {
  const dataW = dataMaxX - dataMinX || 1e-9
  const dataH = dataMaxY - dataMinY || 1e-9
  const scale = Math.min(pixelBox.w / dataW, pixelBox.h / dataH)
  const usedW = dataW * scale
  const usedH = dataH * scale
  const offsetX = pixelBox.x + (pixelBox.w - usedW) / 2
  const offsetY = pixelBox.y + (pixelBox.h - usedH) / 2
  return {
    scale,
    toPx(x: number, y: number): [number, number] {
      return [offsetX + (x - dataMinX) * scale, offsetY + usedH - (y - dataMinY) * scale]
    },
  }
}

function fontPx(pt: number, widthPx: number): number {
  return (pt * widthPx) / 1152 // see this module's docstring: pt * dpi/72, dpi = widthPx/16
}

function linePx(lw: number, widthPx: number): number {
  return Math.max(0.5, (lw * widthPx) / 1152)
}

function roundRect(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.roundRect(x, y, w, h, r)
}

function outlinedText(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  opts: { font: string; fill: string; align: CanvasTextAlign; baseline: CanvasTextBaseline; outlineWidth: number },
) {
  ctx.font = opts.font
  ctx.textAlign = opts.align
  ctx.textBaseline = opts.baseline
  ctx.lineWidth = opts.outlineWidth
  ctx.strokeStyle = OUTLINE_COLOR
  if (opts.outlineWidth > 0) ctx.strokeText(text, x, y)
  ctx.fillStyle = opts.fill
  ctx.fillText(text, x, y)
}

function outlinedStroke(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, color: string, width: number, glow: boolean) {
  ctx.lineWidth = width + 2
  ctx.strokeStyle = OUTLINE_COLOR
  ctx.shadowBlur = 0
  ctx.stroke()
  ctx.lineWidth = width
  ctx.strokeStyle = color
  ctx.shadowBlur = glow ? width * 2 : 0
  ctx.shadowColor = color
  ctx.stroke()
  ctx.shadowBlur = 0
}

export interface DrawOptions {
  /**
   * Whether to wipe the canvas before drawing. True (the default) suits a
   * dedicated, transparent HUD canvas. Pass false when drawing on top of
   * already-rendered video, which is what the render pipeline does -- there,
   * clearing would erase the frame the HUD is supposed to annotate.
   */
  clear?: boolean
}

export class HudRenderer {
  private rows: AnnotatedRow[]
  private config: RenderConfig
  private glow: boolean

  private speedoBox = { x: 0, y: 0, w: 0, h: 0 }
  private speedoT!: Transform
  private ggT!: Transform
  private minimapT!: Transform
  private minimapPath: [number, number][] = []

  constructor(rows: AnnotatedRow[], config: RenderConfig) {
    this.rows = rows
    this.config = config
    this.glow = config.theme === 'cyberpunk'
    this.layout()
  }

  private layout(): void {
    const { widthPx, heightPx } = this.config
    this.speedoBox = figRectToPixelBox(SPEEDO_RECT, widthPx, heightPx)
    this.speedoT = equalAspectTransform(this.speedoBox, 0, 1, 0, 1)

    const ggBox = figRectToPixelBox(GG_RECT, widthPx, heightPx)
    const { limitG } = this.config
    this.ggT = equalAspectTransform(ggBox, -limitG, limitG, -limitG, limitG)

    const minimapBox = figRectToPixelBox(MINIMAP_RECT, widthPx, heightPx)
    const lons = this.rows.map((r) => r.lon)
    const lats = this.rows.map((r) => r.lat)
    const minLon = lons.length ? Math.min(...lons) : -1
    const maxLon = lons.length ? Math.max(...lons) : 1
    const minLat = lats.length ? Math.min(...lats) : -1
    const maxLat = lats.length ? Math.max(...lats) : 1
    this.minimapT = equalAspectTransform(minimapBox, minLon, maxLon, minLat, maxLat)
    this.minimapPath = this.rows.map((r) => this.minimapT.toPx(r.lon, r.lat))
  }

  private drawPanel(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, rect: readonly [number, number, number, number]) {
    const { widthPx, heightPx } = this.config
    const box = figRectToPixelBox(rect, widthPx, heightPx)
    const radius = 0.02 * Math.min(widthPx, heightPx)
    ctx.fillStyle = PANEL_FACE
    roundRect(ctx, box.x, box.y, box.w, box.h, radius)
    ctx.fill()
    ctx.lineWidth = linePx(1.2, widthPx)
    ctx.strokeStyle = PANEL_EDGE
    ctx.stroke()
  }

  private drawSpeedo(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, row: AnnotatedRow) {
    const { widthPx } = this.config
    const t = this.speedoT
    const cx = 0.5
    const cy = 0.38
    const rad = 0.3

    // Static redline-zone track.
    for (const [lo, hi, color] of [
      [0.0, 0.6, GREEN],
      [0.6, 0.85, YELLOW],
      [0.85, 1.0, RED],
    ] as const) {
      ctx.beginPath()
      const steps = 24
      for (let i = 0; i <= steps; i++) {
        const frac = lo + ((hi - lo) * i) / steps
        const ang = angleForFrac(frac)
        const [px, py] = t.toPx(cx + rad * Math.cos(ang), cy + rad * Math.sin(ang))
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.globalAlpha = 0.2
      ctx.lineWidth = linePx(12, widthPx)
      ctx.strokeStyle = color
      ctx.lineCap = 'butt'
      ctx.stroke()
      ctx.globalAlpha = 1
    }

    // Static tick marks, numeric label on every major tick.
    for (let i = 0; i <= 10; i++) {
      const frac = i / 10
      const ang = angleForFrac(frac)
      const major = i % 2 === 0
      const r0 = rad - (major ? 0.05 : 0.03)
      const [x0, y0] = t.toPx(cx + r0 * Math.cos(ang), cy + r0 * Math.sin(ang))
      const [x1, y1] = t.toPx(cx + (rad + 0.012) * Math.cos(ang), cy + (rad + 0.012) * Math.sin(ang))
      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(x1, y1)
      ctx.globalAlpha = major ? 0.6 : 0.3
      ctx.lineWidth = linePx(major ? 1.6 : 1.0, widthPx)
      ctx.strokeStyle = 'white'
      ctx.stroke()
      ctx.globalAlpha = 1
      if (major) {
        const [lx, ly] = t.toPx(cx + (rad + 0.085) * Math.cos(ang), cy + (rad + 0.085) * Math.sin(ang))
        ctx.globalAlpha = 0.65
        outlinedText(ctx, String(Math.round(frac * this.config.maxExpectedSpeedKmh)), lx, ly, {
          font: `${fontPx(8, widthPx)}px sans-serif`,
          fill: 'white',
          align: 'center',
          baseline: 'middle',
          outlineWidth: 0,
        })
        ctx.globalAlpha = 1
      }
    }

    // Dynamic accent arc + needle, colour-matched to the current zone.
    const v = row.speed
    const r = Math.min(v / this.config.maxExpectedSpeedKmh, 1.0)
    const color = r < 0.6 ? GREEN : r < 0.85 ? YELLOW : RED
    ctx.beginPath()
    const steps = Math.max(1, Math.round(r * 100))
    for (let i = 0; i <= steps; i++) {
      const frac = (r * i) / steps
      const ang = angleForFrac(frac)
      const [px, py] = t.toPx(cx + rad * Math.cos(ang), cy + rad * Math.sin(ang))
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.lineCap = 'round'
    outlinedStroke(ctx, color, linePx(4, widthPx), this.glow)

    const ang = angleForFrac(r)
    const tip = rad - 0.055
    const [nx0, ny0] = t.toPx(cx, cy)
    const [nx1, ny1] = t.toPx(cx + tip * Math.cos(ang), cy + tip * Math.sin(ang))
    ctx.beginPath()
    ctx.moveTo(nx0, ny0)
    ctx.lineTo(nx1, ny1)
    ctx.lineCap = 'round'
    outlinedStroke(ctx, 'white', linePx(2.4, widthPx), false)

    // Hub.
    const [hx, hy] = t.toPx(cx, cy)
    ctx.beginPath()
    ctx.arc(hx, hy, 0.026 * t.scale, 0, Math.PI * 2)
    ctx.fillStyle = '#12141a'
    ctx.fill()
    ctx.lineWidth = linePx(1.3, widthPx)
    ctx.strokeStyle = 'white'
    ctx.stroke()

    const [tcx, tcy] = t.toPx(cx, cy - 0.15)
    outlinedText(ctx, String(Math.trunc(v)), tcx, tcy, {
      font: `bold ${fontPx(32, widthPx)}px sans-serif`,
      fill: 'white',
      align: 'center',
      baseline: 'middle',
      outlineWidth: linePx(3, widthPx),
    })
    const [ucx, ucy] = t.toPx(cx, cy - 0.235)
    outlinedText(ctx, 'KM/H', ucx, ucy, {
      font: `bold ${fontPx(10, widthPx)}px sans-serif`,
      fill: GREEN,
      align: 'center',
      baseline: 'middle',
      outlineWidth: linePx(3, widthPx),
    })

    if (this.config.enableLapInfo) {
      const [lapX, lapY] = t.toPx(0.03, 0.95)
      outlinedText(ctx, `LAP ${Math.trunc(row.lap)}`, lapX, lapY, {
        font: `bold ${fontPx(14, widthPx)}px sans-serif`,
        fill: 'cyan',
        align: 'left',
        baseline: 'middle',
        outlineWidth: linePx(3, widthPx),
      })

      const [chronoX, chronoY] = t.toPx(0.97, 0.95)
      outlinedText(ctx, formatLapTime(row.lap_elapsed_s), chronoX, chronoY, {
        font: `bold ${fontPx(12, widthPx)}px sans-serif`,
        fill: 'yellow',
        align: 'right',
        baseline: 'middle',
        outlineWidth: linePx(3, widthPx),
      })

      if (row.last_lap_s > 0 && row.lap_elapsed_s < LAST_LAP_PROMPT_SECONDS) {
        const [llX, llY] = t.toPx(0.97, 0.87)
        outlinedText(ctx, `LAST LAP ${formatLapTime(row.last_lap_s)}`, llX, llY, {
          font: `bold ${fontPx(11, widthPx)}px sans-serif`,
          fill: 'white',
          align: 'right',
          baseline: 'middle',
          outlineWidth: linePx(3, widthPx),
        })
      }
    }
  }

  private drawGg(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, rowIndex: number) {
    const { widthPx, ggTrailFrames } = this.config
    const t = this.ggT
    const row = this.rows[rowIndex]

    const circle = (r: number, alpha: number, dashed: boolean) => {
      const [cx, cy] = t.toPx(0, 0)
      ctx.beginPath()
      ctx.arc(cx, cy, r * t.scale, 0, Math.PI * 2)
      ctx.setLineDash(dashed ? [linePx(4, widthPx), linePx(4, widthPx)] : [])
      ctx.globalAlpha = alpha
      ctx.strokeStyle = 'white'
      ctx.lineWidth = linePx(1, widthPx)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1
    }
    circle(0.5, 0.25, true)
    circle(1.0, 0.4, false)

    const [ax0, ay0] = t.toPx(-this.config.limitG, 0)
    const [ax1, ay1] = t.toPx(this.config.limitG, 0)
    ctx.beginPath()
    ctx.moveTo(ax0, ay0)
    ctx.lineTo(ax1, ay1)
    ctx.globalAlpha = 0.1
    ctx.strokeStyle = 'white'
    ctx.lineWidth = linePx(1, widthPx)
    ctx.stroke()
    const [bx0, by0] = t.toPx(0, -this.config.limitG)
    const [bx1, by1] = t.toPx(0, this.config.limitG)
    ctx.beginPath()
    ctx.moveTo(bx0, by0)
    ctx.lineTo(bx1, by1)
    ctx.stroke()
    ctx.globalAlpha = 1

    const start = Math.max(0, rowIndex - ggTrailFrames)
    ctx.beginPath()
    for (let i = start; i <= rowIndex; i++) {
      const [px, py] = t.toPx(this.rows[i].lat_g, this.rows[i].lon_g)
      if (i === start) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    outlinedStroke(ctx, CYAN, linePx(2, widthPx), this.glow)

    const gVal = Math.hypot(row.lat_g, row.lon_g)
    const ballColor = gVal > 1.0 ? 'red' : gVal > 0.5 ? 'yellow' : GREEN
    const [gx, gy] = t.toPx(row.lat_g, row.lon_g)
    ctx.beginPath()
    ctx.arc(gx, gy, linePx(11, widthPx) / 1.5, 0, Math.PI * 2)
    ctx.fillStyle = ballColor
    ctx.fill()
    ctx.lineWidth = linePx(1, widthPx)
    ctx.strokeStyle = 'white'
    ctx.stroke()

    const [textX, textY] = t.toPx(-this.config.limitG * 0.9, this.config.limitG * 0.7)
    outlinedText(ctx, `${gVal.toFixed(2)} G`, textX, textY, {
      font: `bold ${fontPx(10, widthPx)}px sans-serif`,
      fill: 'white',
      align: 'left',
      baseline: 'middle',
      outlineWidth: linePx(3, widthPx),
    })
  }

  private drawMinimap(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, rowIndex: number) {
    const { widthPx, minimapTrailFrames } = this.config
    const t = this.minimapT

    ctx.beginPath()
    this.minimapPath.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)))
    ctx.globalAlpha = 0.35
    ctx.strokeStyle = CYAN
    ctx.lineWidth = linePx(2, widthPx)
    ctx.stroke()
    ctx.globalAlpha = 1

    const start = Math.max(0, rowIndex - minimapTrailFrames)
    ctx.beginPath()
    for (let i = start; i <= rowIndex; i++) {
      const [px, py] = this.minimapPath[i]
      if (i === start) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    outlinedStroke(ctx, GREEN, linePx(3, widthPx), this.glow)

    const [dx, dy] = this.minimapPath[rowIndex] ?? t.toPx(0, 0)
    ctx.beginPath()
    ctx.arc(dx, dy, linePx(8, widthPx) / 1.5, 0, Math.PI * 2)
    ctx.fillStyle = 'white'
    ctx.lineWidth = linePx(2, widthPx)
    ctx.strokeStyle = 'red'
    ctx.fill()
    ctx.stroke()
  }

  /** Finds the row nearest `timeSec` (this.rows is time-ordered, since it's
   * a windowed slice of already time-ordered telemetry) and draws that
   * frame -- the natural entry point when driving this from a video
   * decoder's own per-frame timestamps, which land on a different, denser
   * or sparser grid than the telemetry's own resampled rate. */
  drawFrameAtTime(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, timeSec: number, options?: DrawOptions): void {
    this.drawFrame(ctx, this.nearestRowIndex(timeSec), options)
  }

  private nearestRowIndex(timeSec: number): number {
    const rows = this.rows
    if (rows.length === 0) return 0
    let lo = 0
    let hi = rows.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (rows[mid].time < timeSec) lo = mid + 1
      else hi = mid
    }
    if (lo > 0 && Math.abs(rows[lo - 1].time - timeSec) <= Math.abs(rows[lo].time - timeSec)) return lo - 1
    return lo
  }

  drawFrame(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D, frameIndex: number, options?: DrawOptions): void {
    const { widthPx, heightPx } = this.config
    // Only when this renderer owns the whole canvas. Compositing onto real
    // footage must pass `clear: false`: the video frame is drawn first, and
    // wiping it here leaves nothing but the HUD floating on an empty canvas.
    if (options?.clear ?? true) ctx.clearRect(0, 0, widthPx, heightPx)
    if (frameIndex >= this.rows.length) return
    const row = this.rows[frameIndex]

    if (this.config.enableSpeedo) {
      this.drawPanel(ctx, SPEEDO_PANEL_RECT)
      this.drawSpeedo(ctx, row)
    }
    if (this.config.enableGg) {
      this.drawPanel(ctx, GG_PANEL_RECT)
      this.drawGg(ctx, frameIndex)
    }
    if (this.config.enableMinimap) {
      this.drawPanel(ctx, MINIMAP_PANEL_RECT)
      this.drawMinimap(ctx, frameIndex)
    }
  }
}

export { DESIGN_WIDTH_PX }
