// SPDX-License-Identifier: GPL-3.0-or-later

import { Channel, invoke } from '@tauri-apps/api/core';
import type { GpuBackendStatusDto } from './generated/GpuBackendStatusDto';
import type { GpuInstallProgressDto } from './generated/GpuInstallProgressDto';

/**
 * Download and unpack the CUDA plugin, reporting byte progress as it goes.
 *
 * The one place the install is invoked from, so the Models settings card and
 * the onboarding setup step cannot drift into two different ideas of what
 * installing means.
 *
 * A finished install does not make the GPU usable yet: ggml builds its backend
 * registry once at startup and has no unload, so the new device only appears
 * on the next launch. The returned status reflects that — it reports the
 * restart, not `ready` — and callers must say so rather than implying the GPU
 * is already in use.
 */
export function installGpuBackend(
  onProgress: (progress: GpuInstallProgressDto) => void,
): Promise<GpuBackendStatusDto> {
  const channel = new Channel<GpuInstallProgressDto>();
  channel.onmessage = onProgress;
  return invoke<GpuBackendStatusDto>('install_gpu_backend', { onProgress: channel });
}
