import { useEffect, useState } from 'react'

/** Uploading, rendering, and downloading all need the coordinator, so this
 * app has no real offline mode -- what the service worker buys is an app
 * shell that still opens (instead of a browser's default connection-error
 * page) when connectivity drops mid-session. Surface that honestly rather
 * than implying offline rendering works. */
export function OfflineBanner() {
  const [online, setOnline] = useState(navigator.onLine)

  useEffect(() => {
    const goOnline = () => setOnline(true)
    const goOffline = () => setOnline(false)
    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    return () => {
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [])

  if (online) return null

  return (
    <div className="offline-banner" role="status">
      You're offline — uploading and rendering need a connection. This screen will keep working once you're back online.
    </div>
  )
}
