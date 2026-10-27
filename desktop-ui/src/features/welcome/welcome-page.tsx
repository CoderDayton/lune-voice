// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import {
  ArrowRight,
  CircleAlert,
  Download,
  Keyboard,
  Mic,
  RefreshCw,
  ShieldCheck,
  Wand2,
  X,
} from 'lucide-react';
import luneLogo from '../../assets/lune-voice-logo.png';
import { formatBytes } from '../../lib/format-bytes';
import { workspaceStore } from '../../store/workspace-store';
import { DownloadChecklist } from '../model-downloads/components/download-checklist';
import { AccelerationStep, useGpuOfferDetail } from './components/acceleration-step';
import { EngineStep } from './components/engine-step';
import { ModelCard } from './components/model-card';
import { ReadyChecks } from './components/ready-checks';
import { ReviewStep } from './components/review-step';
import { WizardStepper } from './components/wizard-stepper';
import {
  type GpuAccelerationOffer,
  isSetupSatisfiable,
  isWizardStep,
  setupDownloadBytes,
  type WizardStep,
} from './setup-logic';
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
    error,
    engineBackend,
    engineWhisper,
    engineParakeet,
    engineLanguage,
    asrLocalName,
    gpuOffer,
    useGpu,
    setUseGpu,
    willInstallGpu,
    gpuDownloadBytes,
    gpuError,
    checklist,
    setEngineBackend,
    setEngineWhisper,
    setEngineParakeet,
    setEngineLanguage,
    steps,
    confirmEngine,
    goNext,
    goBack,
    toggleModel,
    startDownload,
    cancelDownload,
    retry,
  } = setup;

  const requiredOk = isSetupSatisfiable(asrLocalName, models, selected);
  const pending = models.filter((m) => selected.has(m.localName) && !m.ready);
  const runBytes = setupDownloadBytes(models, selected, willInstallGpu ? gpuDownloadBytes : 0);
  const gpuActivating = checklist?.rows.some((r) => r.id === 'gpu' && r.state === 'activating');

  if (phase === 'loading') {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-16">
        <span className="loading loading-spinner loading-lg text-primary" />
        <p className="text-sm text-content-subtle">{t`Checking installed models…`}</p>
      </div>
    );
  }

  return (
    /* Explicit rhythm rather than one uniform gap: header, the stepper, the
       body as one tight block, then the action. Each screen of the wizard is
       one decision, so the whole of it fits the window without scrolling. */
    <div className="flex flex-col">
      <Header
        phase={phase}
        gpuOffer={gpuOffer}
        nothingToInstall={pending.length === 0 && !willInstallGpu}
        gpuActivating={gpuActivating === true}
      />

      {isWizardStep(phase) && (
        <div className="mt-4">
          <WizardStepper steps={steps} current={phase} />
        </div>
      )}

      {/* A floor under the wizard screens, so the header, stepper and buttons
          stay where they were as the user moves between a tall screen and a
          short one; a short screen sits in the middle of it. */}
      <div
        className={`mt-5 flex flex-col gap-4 ${isWizardStep(phase) ? 'min-h-80 justify-center' : ''}`}
      >
        {phase === 'engine' ? (
          <EngineStep
            engineBackend={engineBackend}
            engineWhisper={engineWhisper}
            engineParakeet={engineParakeet}
            engineLanguage={engineLanguage}
            error={error}
            onSelectEngine={setEngineBackend}
            onSelectWhisper={setEngineWhisper}
            onSelectParakeet={setEngineParakeet}
            onSelectLanguage={setEngineLanguage}
          />
        ) : phase === 'acceleration' && gpuOffer ? (
          <AccelerationStep
            offer={gpuOffer}
            useGpu={useGpu}
            downloadBytes={gpuDownloadBytes}
            onToggle={setUseGpu}
          />
        ) : phase === 'extras' ? (
          <div className="grid gap-3 sm:grid-cols-2">
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
                  canToggle={!essential}
                  sizeBytes={m.ready ? null : m.downloadBytes}
                  onToggle={() => toggleModel(m.localName)}
                />
              );
            })}
          </div>
        ) : phase === 'review' ? (
          <ReviewStep
            models={models}
            selected={selected}
            asrLocalName={asrLocalName}
            willInstallGpu={willInstallGpu}
            gpuDownloadBytes={gpuDownloadBytes}
          />
        ) : checklist ? (
          <div className="mx-auto w-full max-w-xl">
            <DownloadChecklist
              checklist={checklist}
              gpuActive={gpuOffer === 'active'}
              gpuError={gpuError}
            />
          </div>
        ) : null}

        {phase === 'done' && (
          <div className="mx-auto flex w-full max-w-xl flex-col gap-4">
            <ReadyChecks />
            <ReadyHint />
          </div>
        )}

        {phase === 'error' && error && (
          <div className="alert alert-error mx-auto w-full max-w-xl gap-2 text-sm" role="alert">
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

      <div className="mt-5">
        <Actions
          phase={phase}
          steps={steps}
          requiredOk={requiredOk}
          pendingCount={pending.length}
          willInstallGpu={willInstallGpu}
          runBytes={runBytes}
          onStart={startDownload}
          onCancel={cancelDownload}
          onRetry={retry}
          onFinish={onFinish}
          onConfirmEngine={confirmEngine}
          onNext={goNext}
          onBack={goBack}
        />
      </div>

      <div className="mt-3">
        <Footer />
      </div>
    </div>
  );
}

interface HeaderProps {
  phase: SetupPhase;
  gpuOffer: GpuAccelerationOffer | null;
  /** Nothing is left to install: every model is here and no GPU step is due. */
  nothingToInstall: boolean;
  /** The GPU plugin is downloaded and loading into the app. */
  gpuActivating: boolean;
}

function Header({ phase, gpuOffer, nothingToInstall, gpuActivating }: HeaderProps) {
  const { t } = useLingui();
  const gpuDetail = useGpuOfferDetail(gpuOffer ?? 'after-download');
  let title: string;
  let subtitle: string;

  if (phase === 'engine') {
    title = t`Welcome to Lune Voice`;
    subtitle = t`Choose your speech engine. Everything runs offline — no cloud, no accounts.`;
  } else if (phase === 'acceleration') {
    title = t`GPU acceleration`;
    subtitle = gpuDetail;
  } else if (phase === 'extras') {
    title = t`Models`;
    subtitle = t`Choose the AI models to install. Everything runs offline — no cloud, no accounts.`;
  } else if (phase === 'review') {
    title = t`Review your setup`;
    subtitle = nothingToInstall
      ? t`All required models are already installed. Continue when you're ready.`
      : t`Here is what setup will install. Everything stays on this machine.`;
  } else if (phase === 'done') {
    title = t`You're all set`;
    subtitle = t`Your models are ready. Everything stays on this machine.`;
  } else if (phase === 'downloading') {
    title = t`Downloading models`;
    subtitle = t`This only happens once. You can keep using your computer in the meantime.`;
  } else if (phase === 'gpu' && gpuActivating) {
    title = t`Turning on your GPU`;
    subtitle = t`GPU support is downloaded. Moving transcription onto it now.`;
  } else if (phase === 'gpu') {
    title = t`Setting up your GPU`;
    subtitle = t`Fetching GPU support. This is the last step, and it only happens once.`;
  } else if (phase === 'error') {
    title = t`Download interrupted`;
    subtitle = t`Check your network and retry. Files that finished downloading are kept.`;
  } else {
    title = t`Welcome to Lune Voice`;
    subtitle = t`Choose your speech engine. Everything runs offline — no cloud, no accounts.`;
  }

  return (
    <header className="flex flex-col items-center gap-2 text-center">
      {/* The mark sits beside the title rather than above it: stacked, it cost
          the screens with the most on them (Parakeet's language picker) the
          last 60px they needed to fit a 720px window. */}
      <div className="flex items-center gap-3">
        {phase === 'done' ? (
          // The finish line is the one place the app can introduce itself, so
          // it shows the product mark rather than one more generic glyph.
          <div className="size-10 overflow-hidden rounded-field ring-1 ring-surface-border-strong">
            <img
              src={luneLogo}
              alt=""
              className="pointer-events-none size-full object-cover"
              draggable={false}
            />
          </div>
        ) : (
          <div className="token-icon size-10">
            <Mic size={20} strokeWidth={1.7} className="text-primary" />
          </div>
        )}
        <h1 className="font-display tracking-heading text-2xl font-semibold text-balance">
          {title}
        </h1>
      </div>
      {/* Wide enough that a two-clause subtitle stays on one line; narrower
          broke it across two ragged lines mid-thought. */}
      <p className="mx-auto max-w-2xl text-balance text-sm text-content-subtle sm:text-base">
        {subtitle}
      </p>
    </header>
  );
}

interface ActionsProps {
  phase: SetupPhase;
  /** The wizard's screens, to tell whether Back has anywhere to go. */
  steps: WizardStep[];
  requiredOk: boolean;
  pendingCount: number;
  willInstallGpu: boolean;
  /** What the run would fetch, GPU included; `null` when unknown. */
  runBytes: number | null;
  onStart: () => void;
  onCancel: () => void;
  onRetry: () => void;
  onFinish: () => void;
  onConfirmEngine: () => void;
  onNext: () => void;
  onBack: () => void;
}

function Actions({
  phase,
  steps,
  requiredOk,
  pendingCount,
  willInstallGpu,
  runBytes,
  onStart,
  onCancel,
  onRetry,
  onFinish,
  onConfirmEngine,
  onNext,
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

  if (phase === 'acceleration' || phase === 'extras') {
    return (
      <div className="flex flex-col items-center gap-2">
        <button type="button" className="btn btn-primary btn-wide" onClick={onNext}>
          {t`Next`}
          <ArrowRight size={16} className="rtl:-scale-x-100" />
        </button>
        <BackAndSkip canGoBack={steps.indexOf(phase) > 0} onBack={onBack} onSkip={onFinish} />
      </div>
    );
  }

  if (phase === 'downloading' || phase === 'gpu') {
    return (
      // No primary button: the checklist above already shows the progress.
      <div className="flex flex-col items-center gap-2">
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
            {t`Please keep the app open until setup finishes`}
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
  const size = runBytes !== null ? formatBytes(runBytes) : null;
  return (
    <div className="flex flex-col items-center gap-2">
      {canStart ? (
        <button type="button" className="btn btn-primary btn-wide" onClick={onStart}>
          <Download size={16} />
          {pendingCount === 0
            ? t`Set up GPU acceleration`
            : size !== null
              ? t`Download ${size}`
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
      <BackAndSkip canGoBack onBack={onBack} onSkip={onFinish} />
    </div>
  );
}

function BackAndSkip({
  canGoBack,
  onBack,
  onSkip,
}: {
  canGoBack: boolean;
  onBack: () => void;
  onSkip: () => void;
}) {
  const { t } = useLingui();
  return (
    <div className="flex items-center gap-2">
      {canGoBack && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onBack}>
          {t`Back`}
        </button>
      )}
      <button type="button" className="btn btn-ghost btn-sm" onClick={onSkip}>
        {t`Skip for now`}
      </button>
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
