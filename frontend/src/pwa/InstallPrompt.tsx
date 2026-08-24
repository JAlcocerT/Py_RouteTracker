import { useEffect, useState } from 'react'

/** Not yet in lib.dom.d.ts. */
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

const DISMISSED_KEY = 'pitlane:install-prompt-dismissed'

function isStandalone(): boolean {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    // iOS Safari's own pre-standard flag
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  )
}

function isIos(): boolean {
  return /iphone|ipad|ipod/i.test(navigator.userAgent)
}

/** A small, dismissible banner offering to install the app. Chrome/Edge/
 * Android fire `beforeinstallprompt`, which we capture and replay from our
 * own button (the browser's default mini-infobar is easy to miss and can't
 * be styled to match the app). iOS Safari never fires that event -- there's
 * no programmatic install API there -- so it gets a one-time instruction
 * card instead. Either way, a dismissal is remembered so this doesn't nag
 * on every visit. */
export function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null)
  const [showIosHint, setShowIosHint] = useState(false)
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(DISMISSED_KEY) === '1')

  useEffect(() => {
    if (isStandalone() || dismissed) return

    const onBeforeInstall = (event: Event) => {
      event.preventDefault()
      setDeferredPrompt(event as BeforeInstallPromptEvent)
    }
    window.addEventListener('beforeinstallprompt', onBeforeInstall)

    const onInstalled = () => {
      setDeferredPrompt(null)
      dismiss()
    }
    window.addEventListener('appinstalled', onInstalled)

    if (isIos()) setShowIosHint(true)

    return () => {
      window.removeEventListener('beforeinstallprompt', onBeforeInstall)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [dismissed])

  function dismiss() {
    localStorage.setItem(DISMISSED_KEY, '1')
    setDismissed(true)
  }

  if (dismissed || (!deferredPrompt && !showIosHint)) return null

  return (
    <div className="install-banner">
      <span className="install-banner__icon" aria-hidden="true">⛽</span>
      <div className="install-banner__body">
        {deferredPrompt ? (
          <>
            <strong>Install PitLane</strong>
            <span>Render straight from your home screen — no browser tab needed.</span>
          </>
        ) : (
          <>
            <strong>Install PitLane</strong>
            <span>
              Tap <em>Share</em>, then <em>Add to Home Screen</em>.
            </span>
          </>
        )}
      </div>
      {deferredPrompt && (
        <button
          className="install-banner__button"
          onClick={async () => {
            await deferredPrompt.prompt()
            const { outcome } = await deferredPrompt.userChoice
            setDeferredPrompt(null)
            if (outcome !== 'accepted') dismiss()
          }}
        >
          Install
        </button>
      )}
      <button className="install-banner__dismiss" aria-label="Dismiss" onClick={dismiss}>
        ✕
      </button>
    </div>
  )
}
