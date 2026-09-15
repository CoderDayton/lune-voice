// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Cpu, Download, Loader2, RefreshCw, Zap } from 'lucide-react';
import { ByteProgress } from '../../../components/ui/byte-progress';
import type { GpuBackendStatusDto } from '../../../lib/backend/generated/GpuBackendStatusDto';
import { formatBytes } from '../../../lib/format-bytes';
import { useGpuBackend } from '../hooks/use-gpu-backend';

/**
 * GPU acceleration, managed alongside the models.
 *
 * The app ships with no NVIDIA code at all — the executable has no load-time
 * dependency on the CUDA runtime, and the bundled CPU backends are enough to
 * run. Turning the GPU on means downloading the optional CUDA plugin, which is
 * a several-hundred-megabyte file that lands on disk next to the models, so it
 * belongs on the same screen as they do.
 */
export function GpuBackendCard() {
  const { t } = useLingui();
  const { status, progress, installing, error, install } = useGpuBackend();

  // Nothing to offer on a build without the native backend, and nothing worth
  // a card before the first status read lands.
  if (!status || status.state === 'unsupported') return null;

  const active = status.state === 'ready';
  // `downloadAvailable` is false on a build whose plugin archive has not been
  // published yet, where the install refuses outright. Offering a
  // several-hundred-megabyte download that can only end in a developer-facing
  // error is worse than offering nothing.
  const offerDownload = status.state === 'not-installed' && status.downloadAvailable;

  const stateClass = active
    ? 'border-success/40 bg-success/5'
    : installing
      ? 'border-primary/50 bg-primary/8'
      : 'border-surface-border bg-surface-inset';

  const iconBg = active
    ? 'bg-success/15 text-success'
    : installing
      ? 'bg-primary/15 text-primary'
      : 'bg-surface-elevated text-content-subtle';

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-medium uppercase tracking-wider text-content-subtle">
        {t`Acceleration`}
      </h3>
      <div className={`rounded-box border p-4 transition-colors ${stateClass}`}>
        <div className="flex items-start gap-3 sm:gap-4">
          <div
            className={`flex size-10 shrink-0 items-center justify-center rounded-field ${iconBg}`}
          >
            {active ? (
              <Check size={18} strokeWidth={2.5} />
            ) : installing ? (
              <span className="loading loading-spinner loading-sm" />
            ) : (
              <Zap size={18} strokeWidth={1.6} />
            )}
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <p className="truncate text-sm font-medium text-content-default sm:text-base">
                {t`GPU acceleration`}
              </p>
              {offerDownload && (
                <span className="text-xs tabular-nums text-content-subtle">
                  {formatBytes(status.downloadBytes)}
                </span>
              )}
              <StatusBadge status={status} />
            </div>

            {/* Progress replaces the description for the whole install, not
                just from the first channel message on: the gap before the
                server answers would otherwise show the download pitch under a
                spinner that is already downloading it. */}
            {installing ? (
              <ByteProgress
                downloadedBytes={progress?.downloadedBytes ?? 0}
                totalBytes={progress?.totalBytes ?? 0}
                fallbackTotalBytes={status.downloadBytes}
              />
            ) : (
              <p className="mt-1 text-xs leading-snug text-content-subtle sm:text-sm">
                <StatusLine status={status} />
              </p>
            )}

            {error && (
              <p className="mt-2 text-sm text-error" role="alert">
                {error}
              </p>
            )}
          </div>

          {offerDownload && (
            <div className="shrink-0">
              <button
                type="button"
                className="btn btn-primary btn-sm gap-1.5 whitespace-nowrap"
                disabled={installing}
                onClick={install}
              >
                {installing ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Download size={13} />
                )}
                {installing ? t`Downloading…` : t`Download`}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: GpuBackendStatusDto }) {
  const { t } = useLingui();

  if (status.state === 'ready') {
    return <span className="badge badge-xs badge-soft badge-success">{t`Active`}</span>;
  }
  if (status.state === 'needs-restart') {
    return <span className="badge badge-xs badge-soft badge-warning">{t`Restart needed`}</span>;
  }
  if (status.state === 'not-installed') {
    return (
      <span className="badge badge-xs badge-soft text-content-subtle">{t`Not installed`}</span>
    );
  }
  return <span className="badge badge-xs badge-soft text-content-subtle">{t`Running on CPU`}</span>;
}

function StatusLine({ status }: { status: GpuBackendStatusDto }): React.ReactNode {
  const { t } = useLingui();

  if (status.state === 'ready') {
    return status.devices.length > 0
      ? t`Transcription is running on ${status.devices.join(', ')}.`
      : t`Transcription is running on your GPU.`;
  }

  if (status.state === 'needs-restart') {
    return (
      <span className="flex items-start gap-1.5">
        <RefreshCw className="mt-0.5 size-3.5 shrink-0" />
        {t`Installed. Restart Lune Voice to start using your GPU.`}
      </span>
    );
  }

  if (status.state === 'driver-missing') {
    return (
      <span className="flex items-start gap-1.5">
        <Cpu className="mt-0.5 size-3.5 shrink-0" />
        {t`No NVIDIA driver was found. Install the driver for your GPU and the download will appear here.`}
      </span>
    );
  }

  if (status.state === 'not-installed') {
    return status.downloadAvailable
      ? t`Transcription runs on the CPU. An optional download moves it to your NVIDIA GPU for noticeably lower latency — most of it is NVIDIA's CUDA maths libraries, which cannot be redistributed in a smaller form.`
      : t`Transcription runs on the CPU. No GPU download has been published for this build yet.`;
  }

  // `failed` — in practice a driver older than the plugin's CUDA runtime. The
  // backend attaches the specifics, which are worth more than a generic line.
  return status.reason ?? t`GPU support is installed but not working.`;
}
