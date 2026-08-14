import { defineConfig } from 'vitest/config'

// Pure-logic unit tests (lib/format.ts) -- no DOM, no ClickHouse, no dev server.
export default defineConfig({
  test: {
    environment: 'node',
  },
})
