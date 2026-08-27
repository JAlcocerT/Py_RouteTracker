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

/**
 * Whether this browser can actually decode the codecs action cams
 * (GoPro-style) almost always use: H.264/HEVC video with AAC audio. Class
 * presence (`hasWebCodecsSupport`) isn't enough to predict this -- many
 * Linux distro-packaged Chromium/Chrome builds implement VideoDecoder/
 * AudioDecoder just fine but ship without licensed decoder support for
 * these specific (patent-encumbered) codecs at all, so `isConfigSupported`
 * comes back false for all of them together even though the API itself
 * works. Probing directly, rather than guessing from class presence, is
 * what actually predicts whether rendering GoPro-style footage will work.
 */
export async function checkActionCamCodecSupport(): Promise<{ h264: boolean; hevc: boolean; aac: boolean }> {
  if (!hasWebCodecsSupport()) return { h264: false, hevc: false, aac: false }
  const { canDecodeVideo, canDecodeAudio } = await import('mediabunny')
  const [h264, hevc, aac] = await Promise.all([canDecodeVideo('avc'), canDecodeVideo('hevc'), canDecodeAudio('aac')])
  return { h264, hevc, aac }
}

/** Turns the checks above into one user-facing message, or `null` if
 * rendering should work fine. Ordered from "nothing will decode at all" down
 * to the narrower, more common gaps, so the message names the actual
 * blocker rather than a generic catch-all. */
export async function describeCodecCompatIssue(): Promise<string | null> {
  if (!hasWebCodecsSupport()) {
    return isInsecureContext()
      ? "This page is loaded over an insecure connection, so this browser won't allow video decoding/encoding here -- rendering will fail at the last step. Open this app via HTTPS, or over localhost/127.0.0.1, instead."
      : "This browser doesn't support the video decoding/encoding APIs this app needs -- rendering will fail at the last step. Try a recent Chrome, Edge, or other Chromium-based browser."
  }

  const { h264, hevc, aac } = await checkActionCamCodecSupport()

  if (!h264 && !hevc) {
    return (
      "This browser can't decode H.264 or HEVC video -- the formats action cams almost always record in -- so " +
      'rendering will fail at the last step. This is common on Linux with a distro-packaged Chromium/Chrome build, ' +
      'which often ships without that licensed codec support built in; try Google Chrome or Microsoft Edge ' +
      '(official builds) instead.'
    )
  }
  if (!hevc) {
    return (
      'This browser can decode H.264 but not HEVC (H.265) video. If your camera recorded in HEVC -- common on ' +
      'newer GoPros at higher resolutions/frame rates -- rendering will fail at the last step. Switch your camera ' +
      'to H.264 recording mode, or try Google Chrome or Microsoft Edge (official builds) instead.'
    )
  }
  if (!aac) {
    return (
      "This browser can't decode AAC audio, which almost all action cams use, so rendering will fail at the last " +
      'step. Try Google Chrome or Microsoft Edge (official builds) instead.'
    )
  }
  return null
}
