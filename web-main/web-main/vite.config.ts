import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';
import {fileURLToPath} from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(() => {
  return {
    plugins: [
      react({
        // Use babel instead of oxc for transpilation
        babel: {
          plugins: [],
        },
      }),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:5000',
          changeOrigin: true,
        },
      },
    },
    build: {
      minify: 'esbuild',
      target: 'es2020',
      // Force using rollup instead of rolldown
      rollupOptions: {
        // This ensures rollup is used
      },
    },
    esbuild: {
      legalComments: 'none',
    },
  };
});
