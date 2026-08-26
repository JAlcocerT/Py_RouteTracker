import { defineConfig, minimal2023Preset } from '@vite-pwa/assets-generator/config'

// Same dark brand background (#0a0a12) is used to pad the maskable and
// apple-touch icons instead of the generator's white default -- a white
// square around this glyph on a dark home-screen dock would clash with the
// rest of the app's neon-on-black identity.
const preset = {
  ...minimal2023Preset,
  maskable: {
    ...minimal2023Preset.maskable,
    resizeOptions: { fit: 'contain', background: '#0a0a12' } as const,
  },
  apple: {
    ...minimal2023Preset.apple,
    resizeOptions: { fit: 'contain', background: '#0a0a12' } as const,
  },
}

export default defineConfig({
  preset,
  images: ['pwa-source/icon.svg'],
})
