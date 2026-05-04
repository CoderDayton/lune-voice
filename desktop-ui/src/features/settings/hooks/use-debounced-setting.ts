import { useCallback, useRef, useState } from 'react';
import { saveWorkspaceSetting, type WorkspaceSettings } from '../../../store/workspace-store';

/**
 * Local state + debounced IPC save for text inputs and sliders.
 * Avoids firing a Tauri round-trip on every keystroke/drag frame.
 */
export function useDebouncedSetting<K extends keyof WorkspaceSettings>(
  key: K,
  storeValue: WorkspaceSettings[K],
  delayMs = 400,
): [WorkspaceSettings[K], (value: WorkspaceSettings[K]) => void] {
  const [local, setLocal] = useState(storeValue);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevStoreRef = useRef(storeValue);

  // Sync from store when the store value changes externally
  if (prevStoreRef.current !== storeValue) {
    prevStoreRef.current = storeValue;
    setLocal(storeValue);
  }

  const update = useCallback(
    (value: WorkspaceSettings[K]) => {
      setLocal(value);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        void saveWorkspaceSetting(key, value);
      }, delayMs);
    },
    [key, delayMs],
  );

  return [local, update];
}
