import { useLingui } from '@lingui/react/macro';
import {
  ArrowRight,
  CircleAlert,
  Download,
  Loader2,
  Mic,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Wand2,
  Zap,
} from 'lucide-react';
import { ByteProgress } from '../../components/ui/byte-progress';
import type { GpuInstallProgressDto } from '../../lib/backend/generated/GpuInstallProgressDto';
import { EngineStep } from './components/engine-step';
import { ModelCard } from './components/model-card';
import { isSetupSatisfiable } from './setup-logic';
import { type SetupPhase, useModelSetup } from './use-model-setup';

interface WelcomePageProps {
  onFinish: () => void;
}

function modelIcon(localName: string) {
  return localName.includes('asr') || localName.includes('whisper') ? Mic : Wand2;
}

export function WelcomePage({ onFinish }: WelcomePageProps) {
  const { t } = useLingui();
  const setup = useModelSetup({ onFinish });

  const {
    phase,
    models,
    selected,
    currentModel,
    progress,
    error,
    completed,
    totalToDownload,
    engineBackend,
    engineWhisper,
    asrLocalName,
    gpuOffer,
    useGpu,
    setUseGpu,
    willInstallGpu,
    gpuProgress,
    gpuDownloadBytes,
    gpuInstalled,
    gpuError,
    setEngineBackend,
    setEngineWhisper,
    confirmEngine,
    backToEngine,
    toggleModel,
    startDownload,
    retry,
  } = setup;

  const requiredOk = isSetupSatisfiable(asrLocalName, models, selected);
  const pending = models.filter((m) => selected.has(m.localName) && !m.ready);
  const overallPct = totalToDownload > 0 ? Math.round((completed / totalToDownload) * 100) : 0;

  if (phase === 'loading') {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-16">
        <span className="loading loading-spinner loading-lg text-primary" />
        <p className="text-sm text-content-subtle">{t`Checking installed models…`}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Header phase={phase} requiredOk={requiredOk} pendingCount={pending.length} error={error} />

      {phase === 'downloading' && (
        <OverallProgress completed={completed} total={totalToDownload} pct={overallPct} />
      )}

      {phase === 'engine' ? (
        <EngineStep
          engineBackend={engineBackend}
          engineWhisper={engineWhisper}
          error={error}
          gpuOffer={gpuOffer}
          useGpu={useGpu}
          gpuDownloadBytes={gpuDownloadBytes}
          onSelectEngine={setEngineBackend}
          onSelectWhisper={setEngineWhisper}
          onToggleGpu={setUseGpu}
        />
      ) : (
        <div className="flex flex-col gap-3">
          {models.map((m) => {
            const essential = m.required || m.localName === asrLocalName;
            return (
              <ModelCard
                key={m.localName}
                name={m.name}
                description={m.description}
                required={essential}
                icon={modelIcon(m.localName)}
                ready={m.ready}
                selected={selected.has(m.localName)}
                active={currentModel === m.localName}
                canToggle={!essential && phase === 'choose'}
                progress={currentModel === m.localName ? progress : null}
                onToggle={() => toggleModel(m.localName)}
              />
            );
          })}
        </div>
      )}

      {(phase === 'gpu' || gpuInstalled || gpuError) && (
        <GpuSetupCard
          installing={phase === 'gpu'}
          progress={gpuProgress}
          totalBytes={gpuDownloadBytes}
          installed={gpuInstalled}
          error={gpuError}
        />
      )}

      {phase === 'error' && error && (
        <div className="alert alert-error gap-2 text-sm" role="alert">
          <CircleAlert size={18} className="shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="font-medium">{t`Download failed`}</p>
            <p className="text-xs opacity-80 break-words">{error}</p>
          </div>
        </div>
      )}

      <Actions
        phase={phase}
        requiredOk={requiredOk}
        pendingCount={pending.length}
        willInstallGpu={willInstallGpu}
        onStart={startDownload}
        onRetry={retry}
        onFinish={onFinish}
        onConfirmEngine={confirmEngine}
        onBack={backToEngine}
      />

      <Footer />
    </div>
  );
}

interface HeaderProps {
  phase: SetupPhase;
  requiredOk: boolean;
  pendingCount: number;
  error: string | null;
}

function Header({ phase, requiredOk, pendingCount, error: _error }: HeaderProps) {
  const { t } = useLingui();
  let title: string;
  let subtitle: string;

  if (phase === 'engine') {
    title = t`Welcome to Lune Voice`;
    subtitle = t`Choose your speech engine. Everything runs offline — no cloud, no accounts.`;
  } else if (phase === 'done') {
    title = t`You're all set`;
    subtitle = t`Your models are ready. Everything stays on this machine.`;
  } else if (phase === 'downloading') {
    title = t`Downloading models`;
    subtitle = t`This only happens once. You can keep using your computer in the meantime.`;
  } else if (phase === 'gpu') {
    title = t`Setting up your GPU`;
    subtitle = t`Fetching the NVIDIA runtime. This is the last step, and it only happens once.`;
  } else if (phase === 'error') {
    title = t`Download interrupted`;
    subtitle = t`Check your network and we'll pick up where we left off.`;
  } else if (!requiredOk) {
    title = t`Welcome to Lune Voice`;
    subtitle = t`Choose the AI models to install. Everything runs offline — no cloud, no accounts.`;
  } else if (pendingCount === 0) {
    title = t`Welcome back`;
    subtitle = t`All required models are already installed. Continue when you're ready.`;
  } else {
    title = t`Welcome to Lune Voice`;
    subtitle = t`Choose the AI models to install. Everything runs offline — no cloud, no accounts.`;
  }

  return (
    <header className="flex flex-col items-center gap-3 text-center">
      <div className="token-icon size-14 sm:size-16">
        {phase === 'done' ? (
          <Sparkles size={26} strokeWidth={1.6} className="text-primary" />
        ) : (
          <Mic size={26} strokeWidth={1.6} className="text-primary" />
        )}
      </div>
      <div>
        <h1 className="font-display tracking-heading text-2xl sm:text-3xl font-semibold text-balance">
          {title}
        </h1>
        <p className="mt-2 text-sm sm:text-base text-content-subtle text-balance max-w-md mx-auto">
          {subtitle}
        </p>
      </div>
    </header>
  );
}

interface GpuSetupCardProps {
  installing: boolean;
  progress: GpuInstallProgressDto | null;
  totalBytes: number;
  installed: boolean;
  error: string | null;
}

/**
 * The CUDA plugin step: its own card rather than a row in the model list,
 * because it is not a model and — unlike every model here — finishing it does
 * not make the GPU usable until the app restarts.
 */
function GpuSetupCard({ installing, progress, totalBytes, installed, error }: GpuSetupCardProps) {
  const { t } = useLingui();

  const tone = error
    ? 'border-warning/40 bg-warning/5'
    : installed
      ? 'border-success/40 bg-success/5'
      : 'border-primary/50 bg-primary/8';

  return (
    <div className={`rounded-box border p-4 ${tone}`}>
      <div className="flex items-start gap-3">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-field bg-primary/15 text-primary">
          {installing ? (
            <Loader2 size={18} className="animate-spin" />
          ) : installed ? (
            <RefreshCw size={18} strokeWidth={1.8} />
          ) : (
            <Zap size={18} strokeWidth={1.8} />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-content-strong">{t`GPU acceleration`}</p>
          <p className="mt-0.5 text-xs text-content-subtle">
            {installing
              ? t`Downloading NVIDIA's CUDA runtime.`
              : installed
                ? t`Installed. Restart Lune Voice to start using your GPU.`
                : t`Could not download GPU support — transcription will run on the CPU. You can retry under Settings › Models.`}
          </p>
          {installing && (
            <ByteProgress
              downloadedBytes={progress?.downloadedBytes ?? 0}
              totalBytes={progress?.totalBytes ?? 0}
              fallbackTotalBytes={totalBytes}
            />
          )}
          {error && <p className="mt-2 break-words text-xs opacity-80">{error}</p>}
        </div>
      </div>
    </div>
  );
}

interface OverallProgressProps {
  completed: number;
  total: number;
  pct: number;
}

function OverallProgress({ completed, total, pct }: OverallProgressProps) {
  const { t } = useLingui();
  return (
    <div className="rounded-box border border-surface-border bg-surface-panel p-4">
      <div className="mb-2 flex items-center justify-between text-xs text-content-subtle">
        <span className="font-medium uppercase tracking-wider">{t`Overall progress`}</span>
        <span className="tabular-nums">
          {completed} / {total} · {pct}%
        </span>
      </div>
      <progress className="progress progress-primary w-full h-2" value={completed} max={total} />
    </div>
  );
}

interface ActionsProps {
  phase: SetupPhase;
  requiredOk: boolean;
  pendingCount: number;
  /** Whether starting the download step also fetches the CUDA plugin. */
  willInstallGpu: boolean;
  onStart: () => void;
  onRetry: () => void;
  onFinish: () => void;
  onConfirmEngine: () => void;
  onBack: () => void;
}

function Actions({
  phase,
  requiredOk,
  pendingCount,
  willInstallGpu,
  onStart,
  onRetry,
  onFinish,
  onConfirmEngine,
  onBack,
}: ActionsProps) {
  const { t } = useLingui();

  if (phase === 'engine') {
    return (
      <div className="flex flex-col items-center gap-2">
        <button type="button" className="btn btn-primary btn-wide" onClick={onConfirmEngine}>
          {t`Continue`}
          <ArrowRight size={16} />
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onFinish}>
          {t`Skip for now`}
        </button>
      </div>
    );
  }

  if (phase === 'downloading' || phase === 'gpu') {
    return (
      <div className="flex flex-col items-center gap-2">
        <button type="button" className="btn btn-primary btn-wide" disabled>
          <Loader2 size={16} className="animate-spin" />
          {t`Downloading…`}
        </button>
        <p className="text-xs text-content-subtle">
          {t`Please keep the app open until downloads finish`}
        </p>
      </div>
    );
  }

  if (phase === 'error') {
    return (
      <div className="flex flex-col gap-2 sm:flex-row sm:justify-center">
        <button type="button" className="btn btn-primary btn-wide" onClick={onRetry}>
          <RefreshCw size={16} />
          {t`Retry download`}
        </button>
        <button type="button" className="btn btn-soft" onClick={onFinish}>
          {t`Skip for now`}
        </button>
      </div>
    );
  }

  if (phase === 'done') {
    return (
      <button type="button" className="btn btn-primary btn-wide self-center" onClick={onFinish}>
        {t`Enter Lune Voice`}
        <ArrowRight size={16} />
      </button>
    );
  }

  // choose — the GPU plugin is downloaded by the same step, so it alone is
  // enough to have something to start even when every model is already here.
  const canStart = requiredOk && (pendingCount > 0 || willInstallGpu);
  return (
    <div className="flex flex-col items-center gap-2">
      {canStart ? (
        <button type="button" className="btn btn-primary btn-wide" onClick={onStart}>
          <Download size={16} />
          {pendingCount === 0
            ? t`Set up GPU acceleration`
            : pendingCount === 1
              ? t`Download 1 model`
              : t`Download ${pendingCount} models`}
        </button>
      ) : !requiredOk ? (
        <button type="button" className="btn btn-primary btn-wide" disabled>
          {t`Select required models`}
        </button>
      ) : (
        <button type="button" className="btn btn-primary btn-wide" onClick={onFinish}>
          {t`Continue`}
          <ArrowRight size={16} />
        </button>
      )}
      <div className="flex items-center gap-2">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onBack}>
          {t`Change engine`}
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onFinish}>
          {t`Skip for now`}
        </button>
      </div>
    </div>
  );
}

function Footer() {
  const { t } = useLingui();
  return (
    <p className="flex items-center justify-center gap-1.5 text-xs text-content-subtle">
      <ShieldCheck size={12} strokeWidth={1.8} />
      {t`Models are stored locally. No telemetry, no accounts.`}
    </p>
  );
}
