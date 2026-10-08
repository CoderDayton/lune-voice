// SPDX-License-Identifier: GPL-3.0-or-later

import { useHotkey } from '@tanstack/react-hotkeys';
import { useRouter } from '@tanstack/react-router';
import { backendClient } from '../../../lib/backend/client';
import { workspaceActions } from '../../../store/workspace-store';
import { navigationShortcuts, quickActionShortcuts } from '../config/navigation';

/**
 * Global keyboard shortcuts for in-app navigation. Bindings come from
 * `navigationShortcuts` so they always match the hints the sidebar renders.
 */
export function useAppShortcuts() {
  const router = useRouter();

  useHotkey(navigationShortcuts['/'], () => {
    void router.navigate({ to: '/' });
  });
  useHotkey(navigationShortcuts['/history'], () => {
    void router.navigate({ to: '/history' });
  });
  useHotkey(navigationShortcuts['/dictionary'], () => {
    void router.navigate({ to: '/dictionary' });
  });
  useHotkey(navigationShortcuts['/snippets'], () => {
    void router.navigate({ to: '/snippets' });
  });
  useHotkey(navigationShortcuts['/styles'], () => {
    void router.navigate({ to: '/styles' });
  });
  useHotkey(navigationShortcuts['/settings'], () => {
    void router.navigate({ to: '/settings' });
  });

  // More menu. Settings' Dictation tab is where the hotkeys live, so the
  // "Keyboard shortcuts" action deep-links straight to it.
  useHotkey(quickActionShortcuts.shortcuts, () => {
    void router.navigate({ to: '/settings', search: { tab: 'dictation' } });
  });
  // `ignoreInputs` is opt-in here: the library lets Ctrl/Meta combos through
  // to text fields by default, and Ctrl+Q is destructive — it exits the
  // process outright, bypassing close-to-tray — so a mistyped Ctrl+Q while
  // composing feedback or editing a dictionary entry would discard the draft.
  useHotkey(
    quickActionShortcuts.quit,
    () => {
      void backendClient.quitApp().catch((error: unknown) => {
        workspaceActions.setError(error instanceof Error ? error.message : String(error));
      });
    },
    { ignoreInputs: true },
  );
}
