/** Feature detection for the render pipeline's hard dependency on WebCodecs
 * (VideoDecoder/AudioDecoder). WebCodecs is only exposed in a secure context
 * (HTTPS, or the browser-special-cased localhost/127.0.0.1/file: origins) --
 * plain http:// access, like this app's own documented Tailscale LAN URL
 * (see README's "Accessing it from other devices over Tailscale"), silently
 * loses the whole API instead of erroring up front. That surfaces much
 * later, deep in rendering, as mediabunny reporting every track --
 * regardless of its actual codec -- as 'undecodable_source_codec', which
 * reads like a codec support problem rather than the real cause. */
export function hasWebCodecsSupport(): boolean {
  return typeof VideoDecoder !== 'undefined' && typeof AudioDecoder !== 'undefined'
}

// `self`, not `window` -- the render pipeline (pipeline.ts) runs inside a
// Web Worker (renderWorker.ts), where `window` doesn't exist but `self` does
// and carries the same `isSecureContext` flag as the page that spawned it.
export function isInsecureContext(): boolean {
  return typeof self !== 'undefined' && self.isSecureContext === false
}
