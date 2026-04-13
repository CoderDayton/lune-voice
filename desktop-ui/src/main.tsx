import { HotkeysProvider } from "@tanstack/react-hotkeys";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { I18nProvider } from "@lingui/react";
import { i18n } from "@lingui/core";
import { activateLocale, defaultLocale, locales } from "./i18n";
import { routeTree } from "./routeTree.gen";
import { backendClient } from "./lib/backend/client";
import { workspaceActions } from "./store/workspace-store";
import { ErrorBoundary } from "./components/feedback";
import { applyTheme, getThemePreference } from "./lib/theme";
import "./app.css";

// Apply saved theme before first paint to avoid flash
applyTheme(getThemePreference());

const router = createRouter({
  routeTree,
  defaultPendingComponent: () => (
    <div className="flex items-center justify-center p-8 sm:p-12">
      <span className="loading loading-spinner loading-md text-primary" />
    </div>
  ),
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

// One-shot: load snapshot once, hydrate store + resolve locale before mount
let bootstrapped = false;
async function bootstrap() {
  if (bootstrapped) return;
  bootstrapped = true;
  try {
    const snapshot = await backendClient.loadWorkspaceSnapshot();
    workspaceActions.hydrate(snapshot);
    const lang = snapshot.settings.language?.split(/[-_]/)[0];
    if (lang && lang !== defaultLocale && lang in locales) {
      await activateLocale(lang);
    }
  } catch {
    // Not in Tauri or snapshot unavailable — app-shell will retry
  }
}

function App() {
  return (
    <ErrorBoundary>
      <I18nProvider i18n={i18n}>
        <HotkeysProvider>
          <RouterProvider router={router} />
        </HotkeysProvider>
      </I18nProvider>
    </ErrorBoundary>
  );
}

// Wait for both default locale AND bootstrap before mounting — prevents locale flash.
// Always mount even if bootstrap fails — app-shell will retry the snapshot load.
Promise.all([activateLocale(defaultLocale), bootstrap()]).catch(() => {}).then(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
});
