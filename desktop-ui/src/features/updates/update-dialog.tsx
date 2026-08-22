import { useLingui } from '@lingui/react/macro';
import { CircleAlert, CircleCheck, Download, Loader2, RefreshCw } from 'lucide-react';
import { useEffect, useId, useRef } from 'react';
import { ByteProgress } from '../../components/ui/byte-progress';
import type { UpdatePhaseDto, UpdateStateDto } from '../../lib/backend/types';
import { useUpdateFlow } from './use-update-flow';

/**
 * Phases the user must not be able to walk away from mid-flight. Closing the
 * modal would not stop the download or the install — it would only hide a
 * package being written over the running app, which is the one moment the user
 * needs to be told to wait.
 */
const BUSY_PHASES: ReadonlySet<UpdatePhaseDto> = new Set<UpdatePhaseDto>([
  'downloading',
  'installing',
]);

/**
 * Every update outcome, reported in the app's own modal.
 *
 * Replaces the native message boxes this flow used to open. Those are drawn by
 * the desktop environment, share none of the app's theme or typography, and on
 * Linux can open behind the window that asked for them — which turns "Check for
 * updates" into a control that appears to do nothing.
 *
 * Mounted once by the shell so all three entry points — the tray item, the
 * Settings button and the sidebar quick action — land in the same place, and so
 * an update found while the window was closed is still waiting when it opens.
 */
export function UpdateDialog() {
  const { state, open, install, dismiss, retry } = useUpdateFlow();
  const titleId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    if (!open && dlg.open) dlg.close();
  }, [open]);

  const busy = state != null && BUSY_PHASES.has(state.phase);

  return (
    <dialog
      ref={dialogRef}
      className="modal"
      aria-labelledby={titleId}
      // Esc would otherwise close a modal whose download keeps running behind
      // it. `close()` from the effect above is unaffected — this only refuses
      // the browser's own cancel.
      onCancel={(event) => {
        if (busy) event.preventDefault();
      }}
      onClose={() => {
        if (!busy) dismiss();
      }}
    >
      <div className="modal-box max-w-lg border border-surface-border bg-surface-glass text-content">
        {state && (
          <UpdateBody
            state={state}
            titleId={titleId}
            onInstall={install}
            onDismiss={dismiss}
            onRetryCheck={retry}
          />
        )}
      </div>
      {/* No click-outside dismissal while busy, for the same reason as Esc. */}
      {!busy && (
        <form method="dialog" className="modal-backdrop">
          <button type="submit">close</button>
        </form>
      )}
    </dialog>
  );
}

interface UpdateBodyProps {
  state: UpdateStateDto;
  titleId: string;
  onInstall: () => void;
  onDismiss: () => void;
  onRetryCheck: () => void;
}

function UpdateBody({ state, titleId, onInstall, onDismiss, onRetryCheck }: UpdateBodyProps) {
  const { t } = useLingui();
  const { phase, currentVersion, availableVersion, notes, message } = state;
  const version = availableVersion ?? '';

  const heading: Record<UpdatePhaseDto, string> = {
    idle: '',
    checking: t`Checking for updates`,
    'up-to-date': t`You're up to date`,
    available: t`Update available`,
    downloading: t`Downloading update`,
    installing: t`Installing update`,
    'check-failed': t`Update check failed`,
    'install-failed': t`Update failed`,
  };

  return (
    <>
      <h3 id={titleId} className="flex items-center gap-2 text-lg font-bold text-content-strong">
        <PhaseIcon phase={phase} />
        {heading[phase]}
      </h3>

      {phase === 'checking' && (
        <p className="mt-2 text-sm text-content-muted">
          {/* Closing here only hides the wait — the check keeps running and
              reopens this with its answer, which is what was asked for. */}
          {t`Asking the release server whether anything newer than ${currentVersion} exists.`}
        </p>
      )}

      {phase === 'up-to-date' && (
        <p className="mt-2 text-sm text-content-muted">
          {t`Lune Voice ${currentVersion} is the latest version.`}
        </p>
      )}

      {phase === 'available' && (
        <>
          <p className="mt-2 text-sm text-content-muted">
            {t`Lune Voice ${version} is available. You have ${currentVersion}.`}
          </p>
          <p className="mt-1 text-sm text-content-muted">
            {t`Installing replaces the app and restarts it. Your settings, history and models are untouched.`}
          </p>
          {notes && <ReleaseNotes notes={notes} />}
        </>
      )}

      {phase === 'downloading' && (
        <>
          <p className="mt-2 text-sm text-content-muted">
            {t`Fetching Lune Voice ${version}. Keep the app open until it finishes.`}
          </p>
          <ByteProgress downloadedBytes={state.downloadedBytes} totalBytes={state.totalBytes} />
        </>
      )}

      {phase === 'installing' && (
        <p className="mt-2 text-sm text-content-muted">
          {/* A .deb or .rpm install goes through pkexec, so the system password
              prompt below is expected rather than a sign something went wrong. */}
          {t`Applying Lune Voice ${version}. Your system may ask for your password, and the app will restart on its own.`}
        </p>
      )}

      {(phase === 'check-failed' || phase === 'install-failed') && message && (
        <div role="alert" className="alert alert-error mt-3 py-2 text-sm">
          {/* The backend messages are already user-facing; showing them verbatim
              keeps real detail ("connection refused", "no package to replace")
              instead of a generic failure nobody can act on. */}
          <span className="min-w-0 flex-1 break-words whitespace-pre-line">{message}</span>
        </div>
      )}

      <div className="modal-action">
        <UpdateActions
          phase={phase}
          onInstall={onInstall}
          onDismiss={onDismiss}
          onRetryCheck={onRetryCheck}
        />
      </div>
    </>
  );
}

function PhaseIcon({ phase }: { phase: UpdatePhaseDto }) {
  if (phase === 'checking' || phase === 'downloading' || phase === 'installing') {
    return <Loader2 size={18} className="shrink-0 animate-spin text-primary" />;
  }
  if (phase === 'up-to-date') {
    return <CircleCheck size={18} strokeWidth={1.8} className="shrink-0 text-success" />;
  }
  if (phase === 'check-failed' || phase === 'install-failed') {
    return <CircleAlert size={18} strokeWidth={1.8} className="shrink-0 text-error" />;
  }
  return <Download size={18} strokeWidth={1.8} className="shrink-0 text-primary" />;
}

/**
 * Release notes come from the update manifest — a remote document. Rendered as
 * plain text in a bounded scroller: never as markup, and never tall enough to
 * push the buttons off a small screen.
 */
function ReleaseNotes({ notes }: { notes: string }) {
  const { t } = useLingui();
  return (
    <details className="mt-3">
      <summary className="cursor-pointer text-sm text-content-subtle">{t`What's new`}</summary>
      <div className="mt-2 max-h-48 overflow-y-auto rounded-field border border-surface-border bg-surface-inset p-3">
        <p className="text-xs whitespace-pre-line break-words text-content-muted">{notes}</p>
      </div>
    </details>
  );
}

interface UpdateActionsProps {
  phase: UpdatePhaseDto;
  onInstall: () => void;
  onDismiss: () => void;
  onRetryCheck: () => void;
}

function UpdateActions({ phase, onInstall, onDismiss, onRetryCheck }: UpdateActionsProps) {
  const { t } = useLingui();

  if (phase === 'downloading' || phase === 'installing') {
    return <p className="text-xs text-content-subtle">{t`This only takes a moment.`}</p>;
  }

  if (phase === 'available') {
    return (
      <>
        <button type="button" className="btn btn-soft" onClick={onDismiss}>
          {t`Later`}
        </button>
        <button type="button" className="btn btn-primary gap-2" onClick={onInstall}>
          <Download size={16} strokeWidth={1.8} />
          {t`Install and restart`}
        </button>
      </>
    );
  }

  if (phase === 'check-failed' || phase === 'install-failed') {
    // A failed install keeps the offer, so its retry re-runs the install rather
    // than asking the release server the question it already answered.
    const again = phase === 'install-failed' ? onInstall : onRetryCheck;
    return (
      <>
        <button type="button" className="btn btn-soft" onClick={onDismiss}>
          {t`Close`}
        </button>
        <button type="button" className="btn btn-primary gap-2" onClick={again}>
          <RefreshCw size={16} strokeWidth={1.8} />
          {t`Try again`}
        </button>
      </>
    );
  }

  return (
    <button type="button" className="btn btn-soft" onClick={onDismiss}>
      {t`Close`}
    </button>
  );
}
