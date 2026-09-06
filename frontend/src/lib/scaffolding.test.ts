import { describe, expect, it } from 'vitest'

// Not a real WebCodecs/OffscreenCanvas smoke test -- jsdom doesn't implement
// either, so that has to happen in a real browser (see the render pipeline's
// own manual test pass in Phase 4). This just proves the two libraries the
// whole client-side pipeline depends on actually load and expose the shapes
// we're about to build against.
describe('client-side pipeline scaffolding', () => {
  it('loads mediabunny', async () => {
    const mediabunny = await import('mediabunny')
    expect(typeof mediabunny.Input).toBe('function')
    expect(typeof mediabunny.Output).toBe('function')
  })

  it('loads mp4box', async () => {
    const MP4Box = await import('mp4box')
    expect(typeof MP4Box.createFile).toBe('function')
  })
})
