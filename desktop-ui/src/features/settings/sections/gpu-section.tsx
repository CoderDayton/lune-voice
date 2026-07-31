import { useLingui } from '@lingui/react/macro';
import { AlertTriangle, Check, Cpu, Download, Zap } from 'lucide-react';
import type { GpuBackendStatusDto } from '../../../lib/backend/generated/GpuBackendStatusDto';
import { SettingsBlock, SettingsSection } from '../components/section';
import { useGpuBackend } from '../hooks/use-gpu-backend';

/** Bytes as a rounded MB/GB string, for download sizes and progress. */
function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

/**
 * GPU acceleration setup.
 *
 * The app ships with no NVIDIA code at all — the executable has no load-time
 * dependency on the CUDA runtime, and the CPU backends bundled beside it are
 * enough to run. This section is where the optional CUDA plugin is downloaded,
 * which is the only thing that turns GPU acceleration on.
 */
export function GpuSection() {
  const { t } = useLingui();
  const { status, progress, installing, error, install } = useGpuBackend();

  if (!status || status.state === 'unsupported') return null;

  return (
    <SettingsSection
      title={t`GPU acceleration`}
      description={t`Transcription runs on the CPU by default. If you have an NVIDIA GPU, an optional download moves it to the GPU for noticeably lower latency.`}
      icon={Zap}
    >
      <SettingsBlock label={t`Status`}>
        <StatusRow status={status} />
      </SettingsBlock>

      {status.state === 'not-installed' && (
        <SettingsBlock
          label={t`GPU support`}
          description={t`Most of the download is NVIDIA's CUDA maths libraries, which cannot be redistributed in a smaller form.`}
        >
          <button type="button" className="btn btn-primary" disabled={installing} onClick={install}>
            {installing ? (
              <span className="loading loading-spinner loading-sm" />
            ) : (
              <Download className="size-4" />
            )}
            {installing
              ? t`Downloading…`
              : t`Download GPU support (${formatBytes(status.downloadBytes)})`}
          </button>

          {progress && (
            <div className="mt-space-sm flex flex-col gap-1">
              <progress
                className="progress progress-primary w-full"
                value={progress.downloadedBytes}
                max={progress.totalBytes || status.downloadBytes}
              />
              <span className="text-xs opacity-70">
                {formatBytes(progress.downloadedBytes)} /{' '}
                {formatBytes(progress.totalBytes || status.downloadBytes)}
              </span>
            </div>
          )}

          {error && (
            <p className="mt-space-sm text-sm text-error" role="alert">
              {error}
            </p>
          )}
        </SettingsBlock>
      )}
    </SettingsSection>
  );
}

function StatusRow({ status }: { status: GpuBackendStatusDto }) {
  const { t } = useLingui();

  switch (status.state) {
    case 'ready':
      return (
        <p className="flex items-center gap-2 text-sm">
          <Check className="size-4 text-success" />
          {status.devices.length > 0
            ? t`Running on ${status.devices.join(', ')}.`
            : t`GPU acceleration is active.`}
        </p>
      );
    case 'driver-missing':
      return (
        <p className="flex items-start gap-2 text-sm">
          <Cpu className="mt-0.5 size-4 opacity-70" />
          <span>
            {t`Running on the CPU. No NVIDIA driver was found — install the driver for your GPU and Lune Voice will offer GPU support here.`}
          </span>
        </p>
      );
    case 'failed':
      return (
        <p className="flex items-start gap-2 text-sm" role="alert">
          <AlertTriangle className="mt-0.5 size-4 text-warning" />
          <span>{status.reason ?? t`GPU support is installed but not working.`}</span>
        </p>
      );
    default:
      return (
        <p className="flex items-start gap-2 text-sm">
          <Cpu className="mt-0.5 size-4 opacity-70" />
          {/* The backend may attach an explanation to a non-failed state too;
              dropping it would leave the user with a bare "running on the CPU"
              and no idea why. */}
          <span>{status.reason ?? t`Running on the CPU.`}</span>
        </p>
      );
  }
}
