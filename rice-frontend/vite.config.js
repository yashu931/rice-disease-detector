import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Forward API calls to Flask backend at localhost:5000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/upload': 'http://127.0.0.1:5000'
    }
  }
})
