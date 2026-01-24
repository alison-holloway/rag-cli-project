import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Tauri expects a fixed port
  server: {
    port: 5173,
    strictPort: true,
  },

  // Prevent vite from obscuring Rust errors
  clearScreen: false,

  // Environment variables that start with TAURI_ will be exposed
  envPrefix: ['VITE_', 'TAURI_'],

  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS and Linux
    target: process.env.TAURI_PLATFORM === 'windows' ? 'chrome105' : 'safari13',
    // Produce sourcemaps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,
    // Optimize chunk size
    chunkSizeWarningLimit: 500,
    // Minification
    minify: 'esbuild',
    // Rollup options for better code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          // Split vendor chunks for better caching
          'vendor-react': ['react', 'react-dom'],
          'vendor-markdown': ['react-markdown'],
          'vendor-highlight': ['highlight.js'],
        },
      },
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-markdown', 'highlight.js'],
  },
})
