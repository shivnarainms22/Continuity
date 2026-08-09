import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Walking skeleton: Vite builds this to static files that continuity/api/app.py
// serves directly (single-container story, see ADR-001). The /api proxy below is
// dev-only convenience -- in production the same origin serves both.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
