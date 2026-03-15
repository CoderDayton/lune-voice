import { HotkeysProvider } from "@tanstack/react-hotkeys";
import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nProvider } from "@lingui/react";
import { i18n } from "@lingui/core";
import { activateLocale, defaultLocale, locales } from "./i18n";
import { routeTree } from "./routeTree.gen";
import "./app.css";
import "./styles/shell.css";
import "./styles/typography.css";
import "./styles/topbar.css";
import "./styles/sidebar.css";
import "./styles/auth.css";
import "./styles/history.css";

const queryClient = new QueryClient();
const router = createRouter({ routeTree, context: { queryClient } });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

function App() {
  useEffect(() => {
    // Activate default locale immediately, then override with persisted preference
    activateLocale(defaultLocale).then(async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const snapshot = await invoke<{ settings: { language: string } }>("load_workspace_snapshot");
        const lang = snapshot.settings.language?.split(/[-_]/)[0];
        if (lang && lang !== defaultLocale && lang in locales) {
          await activateLocale(lang);
        }
      } catch {
        // Not in Tauri or snapshot unavailable — keep default
      }
    });
  }, []);

  return (
    <I18nProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        <HotkeysProvider>
          <RouterProvider router={router} />
        </HotkeysProvider>
      </QueryClientProvider>
    </I18nProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
