import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import ReactDOM from 'react-dom/client';
import { DictationPill } from './features/dictation/dictation-pill';
import { activateLocale, defaultLocale } from './i18n';
import './app.css';

function PillApp() {
  return (
    <I18nProvider i18n={i18n}>
      <DictationPill />
    </I18nProvider>
  );
}

/** Pill window geometry, injected by Rust (`create_pill_window`) before any
 *  page script runs so the CSS footprint matches the window from the first
 *  paint. Absent in the browser/mock — the CSS defaults apply. */
declare global {
  interface Window {
    __PILL_GEOM?: { w: number; h: number; r: number; vertical: boolean };
  }
}

/** Mirror the injected geometry into CSS custom properties (consumed by
 *  `.pill-root` and friends) and flag the vertical layout variant. */
function applyPillGeometry() {
  const geom = window.__PILL_GEOM;
  if (!geom) return;
  const rootStyle = document.documentElement.style;
  rootStyle.setProperty('--pill-w', `${geom.w}px`);
  rootStyle.setProperty('--pill-h', `${geom.h}px`);
  rootStyle.setProperty('--pill-r', `${geom.r}px`);
  if (geom.vertical) document.documentElement.classList.add('pill-vertical');
}

activateLocale(defaultLocale)
  .catch(() => {})
  .then(() => {
    document.documentElement.classList.add('pill-window');
    document.body.classList.add('pill-window');
    applyPillGeometry();

    const root = document.getElementById('pill-root');
    if (!root) {
      console.error('[pill] root element not found');
      return;
    }
    ReactDOM.createRoot(root).render(<PillApp />);
  });
