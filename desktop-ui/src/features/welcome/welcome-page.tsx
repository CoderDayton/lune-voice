// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import {
  ArrowRight,
  CircleAlert,
  Download,
  Keyboard,
  Loader2,
  Mic,
  RefreshCw,
  ShieldCheck,
  Wand2,
  X,
  Zap,
} from 'lucide-react';
import luneLogo from '../../assets/lune-voice-logo.png';
import { ByteProgress } from '../../components/ui/byte-progress';
import type { GpuInstallProgressDto } from '../../lib/backend/generated/GpuInstallProgressDto';
import { workspaceStore } from '../../store/workspace-store';
import { EngineStep } from './components/engine-step';
import { ModelCard } from './components/model-card';
import { ReadyChecks } from './components/ready-checks';
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
    engineParakeet,
    engineLanguage,
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
    setEngineParakeet,
    setEngineLanguage,
    confirmEngine,
    backToEngine,
    toggleModel,
    startDownload,
    cancelDownload,
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
    /* Explicit rhythm rather than one uniform gap. Everything used to sit
       `gap-6` apart, which spaced the primary action exactly as far from the
       last card as the cards were from each other — so the stack had no
       shape and nothing marked where the reading ended and the decision
       began. Now: header, a 32px break, the body as one tight list, a 32px
       break, the action. */
    <div className="flex flex-col">
      <Header phase={phase} requiredOk={requiredOk} pendingCount={pending.length} error={error} />

      <div className="mt-8 flex flex-col gap-4">
        {phase === 'downloading' && (
          <OverallProgress completed={completed} total={totalToDownload} pct={overallPct} />
        )}

        {phase === 'engine' ? (
          <EngineStep
            engineBackend={engineBackend}
            engineWhisper={engineWhisper}
            engineParakeet={engineParakeet}
            engineLanguage={engineLanguage}
            error={error}
            gpuOffer={gpuOffer}
            useGpu={useGpu}
            gpuDownloadBytes={gpuDownloadBytes}
            onSelectEngine={setEngineBackend}
            onSelectWhisper={setEngineWhisper}
            onSelectParakeet={setEngineParakeet}
            onSelectLanguage={setEngineLanguage}
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

        {phase === 'done' && (
          <>
            <ReadyChecks />
            <ReadyHint />
          </>
        )}

        {phase === 'error' && error && (
          <div className="alert alert-error gap-2 text-sm" role="alert">
            <CircleAlert size={18} className="shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="font-medium">{t`Download failed`}</p>
              {/* The shell is `select-none`; the backend's own error string is
                  the one thing here a user needs to copy to report a failure. */}
              <p className="text-xs opacity-80 break-words select-text">{error}</p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-8">
        <Actions
          phase={phase}
          requiredOk={requiredOk}
          pendingCount={pending.length}
          willInstallGpu={willInstallGpu}
          onStart={startDownload}
          onCancel={cancelDownload}
          onRetry={retry}
          onFinish={onFinish}
          onConfirmEngine={confirmEngine}
          onBack={backToEngine}
        />
      </div>

      <div className="mt-6">
        <Footer />
      </div>
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
    subtitle = t`Fetching GPU support. This is the last step, and it only happens once.`;
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
    <header className="flex flex-col items-center gap-4 text-center">
      {/* Was size-14 growing to size-16 around a 26px glyph — a large empty
          box with a small mark adrift in it. One size, tighter ratio. */}
      {phase === 'done' ? (
        // The finish line is the one place the app can introduce itself, so it
        // shows the product mark rather than one more generic glyph.
        <div className="size-14 overflow-hidden rounded-field ring-1 ring-surface-border-strong">
          <img
            src={luneLogo}
            alt=""
            className="pointer-events-none size-full object-cover"
            draggable={false}
          />
        </div>
      ) : (
        <div className="token-icon size-14">
          <Mic size={24} strokeWidth={1.7} className="text-primary" />
        </div>
      )}
      <div>
        <h1 className="font-display tracking-heading text-2xl sm:text-3xl font-semibold text-balance">
          {title}
        </h1>
        {/* `max-w-md` inside a 36rem frame broke every two-clause subtitle
            across two ragged lines mid-thought. Widened so `text-balance`
            has room to find an even break. */}
        <p className="mx-auto mt-2 max-w-lg text-balance text-sm text-content-subtle sm:text-base">
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

function GpuSetupCard({ installing, progress, totalBytes, installed, error }: GpuSetupCardProps) {
  const { t } = useLingui();

  // Only the in-flight state tints the card. A finished install is a plain
  // card with a coloured tile and a badge — the same treatment an installed
  // model gets, so the two sit in one stack without one shouting.
  const tone = installing
    ? 'border-primary/50 bg-primary/8'
    : 'border-surface-border bg-surface-panel';

  const tile = error
    ? 'bg-warning/15 text-warning'
    : installed
      ? 'bg-success/15 text-success'
      : 'bg-primary/15 text-primary';

  return (
    <div className={`rounded-box border p-4 ${tone}`}>
      <div className="flex items-start gap-3">
        <span className={`setup-tile ${tile}`}>
          {installing ? (
            <Loader2 size={18} className="animate-spin" />
          ) : installed ? (
            <RefreshCw size={18} strokeWidth={1.8} />
          ) : (
            <Zap size={18} strokeWidth={1.8} />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-content-strong">{t`GPU acceleration`}</p>
            {installed && !installing && (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            )}
          </div>
          {/* `select-text` because this line can carry the backend's raw error
              string, and the shell is `select-none`. */}
          <p className="mt-0.5 text-xs leading-snug text-content-subtle select-text">
            {installing
              ? t`Downloading GPU support.`
              : installed
                ? // The badge beside the title already says "Installed", so
                  // the sentence only has to carry the part the badge cannot.
                  t`Restart Lune Voice to start using your GPU.`
                : // The backend phrases its own failures for the user and each
                  // one already says transcription continues on the CPU, so
                  // rendering this line *and* the error stated it twice.
                  (error ??
                  t`Could not download GPU support — transcription will run on the CPU. You can retry under Settings › Models.`)}
          </p>
          {installing && (
            <ByteProgress
              downloadedBytes={progress?.downloadedBytes ?? 0}
              totalBytes={progress?.totalBytes ?? 0}
              fallbackTotalBytes={totalBytes}
            />
          )}
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
    <div className="setup-card p-4">
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
  willInstallGpu: boolean;
  onStart: () => void;
  onCancel: () => void;
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
  onCancel,
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
          <ArrowRight size={16} className="rtl:-scale-x-100" />
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
        {/* Only the model queue can be stopped. The GPU plugin install is a
            single archive fetch with no cancel path, and offering a button that
            did nothing would be worse than not offering one. */}
        {phase === 'downloading' ? (
          <button type="button" className="btn btn-ghost btn-sm gap-1.5" onClick={onCancel}>
            <X size={14} strokeWidth={2} />
            {t`Cancel download`}
          </button>
        ) : (
          <p className="text-xs text-content-subtle">
            {t`Please keep the app open until downloads finish`}
          </p>
        )}
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
      <div className="flex flex-col items-center">
        <button type="button" className="btn btn-primary btn-wide" onClick={onFinish}>
          {t`Enter Lune Voice`}
          <ArrowRight size={16} className="rtl:-scale-x-100" />
        </button>
      </div>
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
          <ArrowRight size={16} className="rtl:-scale-x-100" />
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

/**
 * The one thing setup never told anyone: how to actually dictate.
 *
 * Everything before this is downloads. Without the hotkey, finishing setup
 * leaves the user in an app whose entire purpose is triggered by a key
 * combination they have not been shown — the most common way a first run ends
 * in "nothing happens".
 *
 * Reads the live setting rather than hardcoding the default, so it stays true
 * for anyone who has already rebound it.
 */
function ReadyHint() {
  const { t } = useLingui();
  const hotkey = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const activation = useStore(workspaceStore, (s) => s.settings.activationMode);

  return (
    <div className="setup-card p-4">
      <div className="flex items-start gap-3">
        <span className="setup-tile bg-primary/15 text-primary">
          <Keyboard size={18} strokeWidth={1.8} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-content-strong">{t`How to dictate`}</p>
          <p className="mt-1 text-xs leading-snug text-content-subtle">
            {activation === 'Hold'
              ? t`Hold your hotkey, speak, then let go. The text is typed into whatever app you were using.`
              : t`Press your hotkey, speak, then press it again. The text is typed into whatever app you were using.`}
          </p>
          <p className="mt-2 flex items-center gap-2 text-xs text-content-subtle">
            <kbd className="kbd kbd-sm">{hotkey || t`Ctrl+Space`}</kbd>
            <span>{t`— change it any time in Settings › Dictation.`}</span>
          </p>
        </div>
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
