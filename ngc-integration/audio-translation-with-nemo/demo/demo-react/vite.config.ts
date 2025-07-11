import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base:'/demo',
  plugins: [react()],
  define: {
    global: {},
  },
  server: {
    // host: 'https://localhost:54265'
  }
})
