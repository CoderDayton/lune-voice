// SPDX-License-Identifier: GPL-3.0-or-later

import { resolve } from "node:path";
import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import { lingui } from "@lingui/vite-plugin";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig(async () => ({
  plugins: [
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    // plugin-react 6 transforms with oxc and no longer accepts a `babel` option,
    // so the two babel plugins this project needs run through their own pass.
    // The lingui macro MUST be compiled away here — left to run at runtime it
    // throws on import and nothing mounts.
    babel({
      presets: [reactCompilerPreset()],
      plugins: ["@lingui/babel-plugin-lingui-macro"],
    }),
    lingui(),
    tailwindcss(),
  ],
  clearScreen: false,
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        pill: resolve(import.meta.dirname, "pill.html"),
      },
    },
    target: "esnext",
    // vite 8 bundles with rolldown and no longer ships esbuild; `true` picks
    // its own (oxc) minifier.
    minify: true,
  },
  optimizeDeps: {
    include: [
      "react",
      "react-dom",
      "@tanstack/react-router",
      "@tanstack/react-store",

      "@tanstack/react-hotkeys",
      "@lingui/core",
      "@lingui/react",
      "@tauri-apps/api/core",
      "@tauri-apps/api/event",
      "@tauri-apps/api/window",
      "@tauri-apps/api/dpi",
    ],
  },
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
}));
