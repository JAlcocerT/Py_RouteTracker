import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      // 'prompt' (not 'autoUpdate') on purpose: a render runs entirely in
      // this tab (see src/lib/render/pipeline.ts), so a service worker
      // swap that reloads mid-render would abort it outright -- asking the
      // user to reload, rather than doing it silently underneath them, is
      // the whole point here. See src/pwa/UpdateToast.tsx.
      registerType: 'prompt',
      injectRegister: false,
      workbox: {
        // ffmpeg.wasm's core is ~32MB and only ever loads on the
        // software-decode fallback path (see lib/render/wasmTranscode.ts),
        // which most users never hit. Precaching it would push a 32MB
        // download onto every install for a feature they'll likely never
        // use -- so it's excluded here and fetched on demand instead.
        globIgnores: ['**/ffmpeg-core*.wasm'],
        runtimeCaching: [
          {
            // Once a user *has* paid the download, keep it: this path is
            // for people whose browser always needs it, so re-fetching
            // 32MB per render would be a poor trade. CacheFirst is safe
            // because the filename is content-hashed.
            urlPattern: /ffmpeg-core.*\.wasm$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'ffmpeg-core-wasm',
              expiration: { maxEntries: 2 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
      manifest: {
        name: 'PitLane — Telemetry Overlay Studio',
        short_name: 'PitLane',
        description: 'Overlay GPS/G-force telemetry HUDs onto action-cam footage: trim, pick your widgets, render, download.',
        theme_color: '#0a0a12',
        background_color: '#0a0a12',
        display: 'standalone',
        orientation: 'any',
        start_url: '/',
        scope: '/',
        icons: [
          { src: 'pwa-64x64.png', sizes: '64x64', type: 'image/png' },
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'maskable-icon-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  worker: {
    // The render pipeline worker imports mediabunny/mp4box as ES modules.
    format: 'es',
  },
})
