import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      // 'prompt' (not 'autoUpdate') on purpose: this app talks to a
      // coordinator whose API can change shape across releases, so a
      // silently-swapped service worker mid-render would be worse than
      // asking the user to reload -- see src/pwa/UpdateToast.tsx.
      registerType: 'prompt',
      injectRegister: false,
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
      workbox: {
        // Everything under /api/* (uploads, status polls, the finished
        // render download) must always hit the network -- these are large,
        // job-scoped, and frequently-changing; Workbox is left to precache
        // only the built app shell (JS/CSS/HTML/icons) it finds via its
        // default globPatterns, with no runtimeCaching rules added for /api
        // so those requests are never intercepted at all.
        navigateFallbackDenylist: [/^\/api\//],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL ?? 'http://localhost:7000',
        changeOrigin: true,
      },
    },
  },
})
