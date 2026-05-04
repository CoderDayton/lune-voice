import { useHotkey } from '@tanstack/react-hotkeys';
import { useRouter } from '@tanstack/react-router';

/**
 * Global keyboard shortcuts for in-app navigation.
 * Ctrl+1-4 switch pages.
 */
export function useAppShortcuts() {
  const router = useRouter();

  useHotkey('Control+1', () => {
    void router.navigate({ to: '/' });
  });
  useHotkey('Control+2', () => {
    void router.navigate({ to: '/history' });
  });
  useHotkey('Control+3', () => {
    void router.navigate({ to: '/dictionary' });
  });
  useHotkey('Control+4', () => {
    void router.navigate({ to: '/settings' });
  });
}
