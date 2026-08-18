import { useLingui } from '@lingui/react/macro';
import { Check, CircleAlert, type LucideIcon, Mic, Zap } from 'lucide-react';
import { formatBytes } from '../../../lib/format-bytes';
import type { AsrBackendValue, WhisperModelValue } from '../../settings/data/settings-data';
import { whisperModels } from '../../settings/data/settings-data';
import type { GpuAccelerationOffer } from '../setup-logic';

interface EngineStepProps {
  engineBackend: AsrBackendValue;
  engineWhisper: WhisperModelValue;
  error: string | null;
  /** What the GPU acceptance card should offer, or `null` to hide it. */
  gpuOffer: GpuAccelerationOffer | null;
  /** Whether GPU acceleration is currently accepted (mirrors `useGpu`). */
  useGpu: boolean;
  /** Size of the CUDA plugin download setup will make, when it will make one. */
  gpuDownloadBytes: number;
  onSelectEngine: (backend: AsrBackendValue) => void;
  onSelectWhisper: (model: WhisperModelValue) => void;
  onToggleGpu: (next: boolean) => void;
}

interface EngineOption {
  value: Extract<AsrBackendValue, 'WhisperCpp'>;
  label: string;
  tagline: string;
  detail: string;
  icon: LucideIcon;
}

/**
 * First-run engine picker: the two on-device speech engines. The cloud backend
 * is intentionally omitted here — onboarding is about getting a local model on
 * disk; the remote option lives in Settings.
 */
export function EngineStep({
  engineBackend,
  engineWhisper,
  error,
  gpuOffer,
  useGpu,
  gpuDownloadBytes,
  onSelectEngine,
  onSelectWhisper,
  onToggleGpu,
}: EngineStepProps) {
  const { t } = useLingui();

  const engines: EngineOption[] = [
    {
      value: 'WhisperCpp',
      label: t`Whisper`,
      tagline: t`Multilingual · pick a size`,
      detail: t`whisper.cpp running on-device. Choose a size to trade speed for accuracy.`,
      icon: Mic,
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {engines.map((engine) => (
          <EngineCard
            key={engine.value}
            option={engine}
            active={engineBackend === engine.value}
            onSelect={() => onSelectEngine(engine.value)}
          />
        ))}
      </div>

      {engineBackend === 'WhisperCpp' && (
        <fieldset className="rounded-box border border-surface-border bg-surface-panel p-4">
          <legend className="px-1 text-xs font-medium uppercase tracking-wider text-content-subtle">
            {t`Whisper model size`}
          </legend>
          <div className="mt-1 flex flex-col gap-2">
            {whisperModels.map((m) => {
              const active = engineWhisper === m.value;
              return (
                <button
                  key={m.value}
                  type="button"
                  aria-pressed={active}
                  onClick={() => onSelectWhisper(m.value)}
                  className={`flex items-center justify-between gap-3 rounded-field border px-3 py-2.5 text-left transition-colors ${
                    active
                      ? 'border-primary/50 bg-primary/8'
                      : 'border-surface-border bg-surface-elevated hover:border-primary/40 hover:bg-primary/5'
                  }`}
                >
                  <span className="min-w-0">
                    <span className="flex items-center gap-2">
                      <span
                        className={`text-sm font-medium ${active ? 'text-content-strong' : 'text-content'}`}
                      >
                        {m.label}
                      </span>
                      <span className="text-xs tabular-nums text-content-subtle">{m.size}</span>
                    </span>
                    <span className="mt-0.5 block text-xs text-content-subtle">
                      {m.description}
                    </span>
                  </span>
                  {active && (
                    <Check size={16} strokeWidth={2.5} className="shrink-0 text-primary" />
                  )}
                </button>
              );
            })}
          </div>
        </fieldset>
      )}

      {gpuOffer && (
        <label className="flex items-center justify-between gap-3 rounded-box border border-surface-border bg-surface-panel p-4">
          <span className="flex min-w-0 items-start gap-3">
            <span className="flex size-9 shrink-0 items-center justify-center rounded-field bg-primary/15 text-primary">
              <Zap size={18} strokeWidth={1.8} />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-content-strong">
                {t`GPU acceleration`}
              </span>
              <span className="mt-0.5 block text-xs text-content-subtle">
                {/* Promising speed the user does not have yet would be a lie
                    the first dictation exposes, so say which of the three
                    situations this is: already accelerated, one restart away,
                    or a download setup is about to make. */}
                {gpuOffer === 'active'
                  ? t`Your GPU is ready — use it for faster transcription.`
                  : gpuOffer === 'needs-restart'
                    ? t`Installed. Restart Lune Voice to start using your GPU.`
                    : t`Your GPU qualifies. Setup will download the ${formatBytes(gpuDownloadBytes)} NVIDIA runtime, then a restart turns it on.`}
              </span>
            </span>
          </span>
          <input
            type="checkbox"
            className="toggle toggle-primary shrink-0"
            checked={useGpu}
            onChange={(e) => onToggleGpu(e.target.checked)}
            aria-label={t`Use GPU acceleration`}
          />
        </label>
      )}

      {error && (
        <div className="alert alert-error gap-2 text-sm" role="alert">
          <CircleAlert size={18} className="shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
        </div>
      )}
    </div>
  );
}

function EngineCard({
  option,
  active,
  onSelect,
}: {
  option: EngineOption;
  active: boolean;
  onSelect: () => void;
}) {
  const Icon = option.icon;
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`group flex flex-col gap-3 rounded-box border p-4 text-left transition-all duration-150 ${
        active
          ? 'border-primary/50 bg-primary/8 shadow-sm'
          : 'border-surface-border bg-base-200/60 hover:border-primary/45 hover:bg-primary/8'
      }`}
    >
      <div className="flex items-center justify-between">
        <span
          className={`flex size-11 items-center justify-center rounded-field transition-colors ${
            active
              ? 'bg-primary/15 text-primary'
              : 'bg-base-300 text-content-subtle group-hover:text-primary'
          }`}
        >
          <Icon size={20} strokeWidth={1.7} />
        </span>
        <span
          className={`flex size-5 items-center justify-center rounded-full border-2 transition-colors ${
            active ? 'border-primary bg-primary' : 'border-surface-border'
          }`}
          aria-hidden="true"
        >
          {active && <Check size={12} strokeWidth={3} className="text-primary-content" />}
        </span>
      </div>
      <div>
        <div className="flex items-baseline gap-2">
          <span
            className={`font-medium ${active ? 'text-content-strong' : 'text-content-default'}`}
          >
            {option.label}
          </span>
        </div>
        <p className="mt-0.5 text-xs font-medium text-primary/80">{option.tagline}</p>
        <p className="mt-1.5 text-xs leading-snug text-content-subtle">{option.detail}</p>
      </div>
    </button>
  );
}
