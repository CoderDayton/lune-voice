import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { HotkeysProvider } from '@tanstack/react-hotkeys';
import { createRouter, RouterProvider } from '@tanstack/react-router';
import ReactDOM from 'react-dom/client';
import { ErrorBoundary } from './components/feedback';
import { activateLocale, defaultLocale, locales } from './i18n';
import { backendClient } from './lib/backend/client';
import { applyTheme, getThemePreference } from './lib/theme';
import { routeTree } from './routeTree.gen';
import { workspaceActions } from './store/workspace-store';
import './app.css';

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

declare module '@tanstack/react-router' {
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
  } catch (error) {
    console.warn('[bootstrap] workspace snapshot unavailable', error);
    // Not in Tauri or snapshot unavailable — app-shell will retry
  }
}

function App() {
  return (
    // The provider wraps the boundary, not the other way round: the fallback
    // is translated too, and it can only reach the catalog from inside.
    <I18nProvider i18n={i18n}>
      <ErrorBoundary>
        <HotkeysProvider>
          <RouterProvider router={router} />
        </HotkeysProvider>
      </ErrorBoundary>
    </I18nProvider>
  );
}

// Wait for both default locale AND bootstrap before mounting — prevents locale flash.
// Always mount even if bootstrap fails — app-shell will retry the snapshot load.
Promise.all([activateLocale(defaultLocale), bootstrap()])
  .catch((error) => {
    console.warn('[bootstrap] locale or workspace bootstrap failed', error);
  })
  .then(() => {
    const root = document.getElementById('root');
    if (!root) {
      console.error('[bootstrap] root element not found');
      return;
    }
    ReactDOM.createRoot(root).render(<App />);
  });
