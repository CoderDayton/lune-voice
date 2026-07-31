import { Channel, invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useState } from 'react';
import type { GpuBackendStatusDto } from '../../../lib/backend/generated/GpuBackendStatusDto';
import type { GpuInstallProgressDto } from '../../../lib/backend/generated/GpuInstallProgressDto';
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
 * GPU backend lifecycle: reports what acceleration can do on this machine and
 * downloads the CUDA plugin when the user opts in.
 *
 * Unlike the model downloads, a finished install does not become usable
 * immediately — ggml builds its backend registry once at startup and cannot
 * unload it, so the new device only appears after a restart. The backend
 * reflects that by not returning a `ready` status here, and the UI says so
 * rather than implying the GPU is already in use.
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

    const channel = new Channel<GpuInstallProgressDto>();
    channel.onmessage = setProgress;

    invoke<GpuBackendStatusDto>('install_gpu_backend', { onProgress: channel })
      .then(setStatus)
      .catch((e) => setError(String(e)))
      .finally(() => {
        setInstalling(false);
        setProgress(null);
      });
  }, []);

  return { status, progress, installing, error, install, refresh };
}
