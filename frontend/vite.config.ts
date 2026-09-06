import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  optimizeDeps: {
    // libde265 is an emscripten bundle that locates its own .wasm at runtime;
    // Vite's dependency pre-bundler rewrites it in ways that break that, and
    // the package's README asks for this exclusion by name.
    exclude: ['@yume-chan/libde265'],
  },
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
        // The libde265 wasm (~420KB) precaches without needing the size cap
        // raised, but .wasm has to be matched explicitly -- it isn't in
        // workbox's default patterns, and an offline-capable PWA that can't
        // decode HEVC offline would be missing the point.
        globPatterns: ['**/*.{js,css,html,ico,png,svg,wasm}'],
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
