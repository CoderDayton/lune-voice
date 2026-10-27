// SPDX-License-Identifier: GPL-3.0-or-later

import type { ReactNode } from 'react';

/**
 * Full-viewport layout for the first-run welcome screen.
 * Centered content, no sidebar or topbar — escapes the main app shell.
 * Reuses the app backdrop so the surface feels continuous with the rest
 * of the app once setup is complete.
 */
export function WelcomeLayout({ children }: { children: ReactNode }) {
  return (
    <div className="welcome-shell relative h-dvh overflow-hidden select-none">
      <div className="app-shell-backdrop" aria-hidden="true" />
      <div className="welcome-scroll relative z-10 flex h-full w-full items-center justify-center overflow-y-auto px-4 py-6 sm:px-8">
        <div className="welcome-frame w-full max-w-3xl my-auto animate-page-in">{children}</div>
      </div>
    </div>
  );
}
