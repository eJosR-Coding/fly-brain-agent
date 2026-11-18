import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // the live session (/api/ws/live) is a WebSocket
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  preview: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // the live session (/api/ws/live) is a WebSocket
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
