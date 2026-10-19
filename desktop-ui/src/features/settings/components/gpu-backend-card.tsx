// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Cpu, Download, Loader2, RefreshCw, Zap } from 'lucide-react';
import type { ReactNode } from 'react';
import { ByteProgress } from '../../../components/ui/byte-progress';
import type { GpuBackendStatusDto } from '../../../lib/backend/generated/GpuBackendStatusDto';
import { formatBytes } from '../../../lib/format-bytes';
import { useGpuBackend } from '../hooks/use-gpu-backend';

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

  let side: ReactNode = null;
  if (installing) {
    side = (
      <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-primary">
        <Loader2 size={13} className="animate-spin" />
        {t`Downloading…`}
      </span>
    );
  } else if (offerDownload) {
    side = (
      <button
        type="button"
        className="btn btn-sm btn-ghost gap-1.5 border border-surface-border-strong text-content-strong hover:border-primary"
        onClick={install}
      >
        <Download size={13} />
        {t`Download`}
      </button>
    );
  } else {
    side = <StatusTag status={status} />;
  }

  return (
    <div className="mx-4 rounded-box border border-surface-border bg-surface-card px-5 py-4">
      <div className="flex items-center gap-4">
        <Zap
          size={18}
          strokeWidth={1.8}
          className={`shrink-0 ${active ? 'text-success' : 'text-content-muted'}`}
        />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="flex flex-wrap items-baseline gap-x-2 text-[15px] font-medium text-content-strong">
            {t`GPU acceleration`}
            {offerDownload && (
              <span className="text-[13px] font-normal tabular-nums text-content-muted">
                {formatBytes(status.downloadBytes)}
              </span>
            )}
          </span>

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
            <span className="text-[13px] text-content-muted">
              <StatusLine status={status} />
            </span>
          )}
          {error && (
            <p className="mt-1 text-[13px] text-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <span className="flex min-w-24 shrink-0 justify-end">{side}</span>
      </div>
    </div>
  );
}

function StatusTag({ status }: { status: GpuBackendStatusDto }) {
  const { t } = useLingui();

  if (status.state === 'ready') {
    return (
      <span className="inline-flex items-center gap-1 text-[13px] font-medium text-success">
        <Check size={13} strokeWidth={2.5} />
        {t`Active`}
      </span>
    );
  }
  if (status.state === 'needs-restart') {
    return <span className="text-[13px] font-medium text-warning">{t`Restart needed`}</span>;
  }
  if (status.state === 'not-installed') {
    return <span className="text-[13px] text-content-muted">{t`Not installed`}</span>;
  }
  return <span className="text-[13px] text-content-muted">{t`Running on CPU`}</span>;
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
