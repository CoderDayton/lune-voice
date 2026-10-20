// SPDX-License-Identifier: GPL-3.0-or-later

import { invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useState } from 'react';
import type { GpuBackendStatusDto } from '../../../lib/backend/generated/GpuBackendStatusDto';
import type { GpuInstallProgressDto } from '../../../lib/backend/generated/GpuInstallProgressDto';
import { installGpuBackend } from '../../../lib/backend/gpu-install';
import { isTauriRuntime } from '../../../lib/backend/runtime';

export interface GpuBackendState {
  status: GpuBackendStatusDto | null;
  /** Non-null while the archive is downloading. */
  progress: GpuInstallProgressDto | null;
  installing: boolean;
  /** Set when the install failed; cleared when a new one starts. */
  error: string | null;
  install: () => void;
  refresh: () => void;
}

/**
 * A finished install reports `needs-restart` rather than `ready` — ggml builds
 * its backend registry once at startup and has no unload — so the status the
 * install resolves with is the whole answer, and nothing here has to remember
 * that an install happened.
 */
export function useGpuBackend(): GpuBackendState {
  const [status, setStatus] = useState<GpuBackendStatusDto | null>(null);
  const [progress, setProgress] = useState<GpuInstallProgressDto | null>(null);
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!isTauriRuntime()) return;
    invoke<GpuBackendStatusDto>('gpu_backend_status')
      .then(setStatus)
      .catch((e) => console.error('gpu_backend_status:', e));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const install = useCallback(() => {
    if (!isTauriRuntime()) return;
    setInstalling(true);
    setError(null);
    setProgress(null);

    installGpuBackend(setProgress)
      .then(setStatus)
      .catch((e) => setError(String(e)))
      .finally(() => {
        setInstalling(false);
        setProgress(null);
      });
  }, []);

  return { status, progress, installing, error, install, refresh };
}
