import { defineConfig } from 'vitest/config'

export default defineConfig({
  resolve: {
    // Array form with an anchored regex, not the object form: object aliases
    // match by prefix, which would also rewrite the sibling
    // '@yume-chan/libde265/libde265.wasm?url' import into a broken path.
    alias: [
      {
        // The package declares only a `browser` entry point, which Vite
        // honours when building for the browser but not under vitest's Node
        // resolution. The module is mocked in the tests that touch it; this
        // only has to resolve.
        find: /^@yume-chan\/libde265$/,
        replacement: '@yume-chan/libde265/libde265.mjs',
      },
    ],
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
})
