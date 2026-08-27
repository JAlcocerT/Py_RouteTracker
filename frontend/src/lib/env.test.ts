import { afterEach, describe, expect, it, vi } from 'vitest'
import { hasWebCodecsSupport, isInsecureContext } from './env'

describe('hasWebCodecsSupport', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('is false when VideoDecoder/AudioDecoder are unavailable (jsdom, or an insecure-context browser)', () => {
    expect(hasWebCodecsSupport()).toBe(false)
  })

  it('is true once both decoders are present', () => {
    vi.stubGlobal('VideoDecoder', class {})
    vi.stubGlobal('AudioDecoder', class {})
    expect(hasWebCodecsSupport()).toBe(true)
  })
})

describe('isInsecureContext', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('is true when self.isSecureContext is explicitly false', () => {
    vi.stubGlobal('self', { isSecureContext: false })
    expect(isInsecureContext()).toBe(true)
  })

  it('is false when self.isSecureContext is true', () => {
    vi.stubGlobal('self', { isSecureContext: true })
    expect(isInsecureContext()).toBe(false)
  })
})
