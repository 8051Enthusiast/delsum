import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  base: '',
  worker: {
    format: 'es'
  },
  build: {
    target: 'es2022',
    lib: {
      entry: resolve(dirname(fileURLToPath(import.meta.url)), 'src/main.ts'),
      name: "delsum",
      fileName: "delsum",
      formats: ["es"]
    },
  },
})
