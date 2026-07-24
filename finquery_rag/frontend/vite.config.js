import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // The Vite config is executed from the frontend directory by both npm and
  // the deployment launcher, so `.` resolves to its own .env files.
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          // In an SSH-tunnel deployment the browser's 127.0.0.1 is the
          // developer machine, not the server.  Keep the backend target on
          // the Vite host and expose only this same-origin proxy to clients.
          target: env.VITE_API_PROXY_TARGET || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
