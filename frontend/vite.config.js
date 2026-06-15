import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_TARGET || `http://127.0.0.1:${env.LOCALTUNE_PORT || 6543}`;

  return {
    plugins: [react()],
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            query: ["@tanstack/react-query"],
            charts: ["recharts"],
            icons: ["lucide-react"],
          },
        },
      },
    },
    server: {
      proxy: {
        "/api": apiTarget,
      },
    },
  };
});
