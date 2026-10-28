import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // the Flask dashboard API (app/server.py) -- avoids CORS entirely
      '/api': 'http://127.0.0.1:5000',
    },
  },
})
