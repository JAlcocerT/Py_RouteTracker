import { useRegisterSW } from 'virtual:pwa-register/react'

/** Registers the service worker and surfaces the two states Workbox's
 * "prompt" flow leaves to the app: a new version waiting to take over
 * (needRefresh), and the first successful install caching everything for
 * offline use (offlineReady). registerType is 'prompt' rather than
 * 'autoUpdate' -- see vite.config.ts -- so both are user-driven, never
 * silent. */
export function UpdateToast() {
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    offlineReady: [offlineReady, setOfflineReady],
    updateServiceWorker,
  } = useRegisterSW({
    onRegisteredSW(_url, registration) {
      // Long-lived tab (a render can run for a while): check for a new
      // build periodically rather than only on full page load.
      if (!registration) return
      window.setInterval(() => registration.update(), 60 * 60 * 1000)
    },
  })

  if (needRefresh) {
    return (
      <div className="pwa-toast" role="alert">
        <span>A new version of PitLane is ready.</span>
        <div className="pwa-toast__actions">
          <button className="pwa-toast__button" onClick={() => updateServiceWorker(true)}>
            Reload
          </button>
          <button className="pwa-toast__dismiss" onClick={() => setNeedRefresh(false)}>
            Later
          </button>
        </div>
      </div>
    )
  }

  if (offlineReady) {
    return (
      <div className="pwa-toast pwa-toast--muted" role="status">
        <span>PitLane is installed and ready to load offline.</span>
        <button className="pwa-toast__dismiss" onClick={() => setOfflineReady(false)}>
          Dismiss
        </button>
      </div>
    )
  }

  return null
}
