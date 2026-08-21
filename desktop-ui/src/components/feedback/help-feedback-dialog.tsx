import { useLingui } from '@lingui/react/macro';
import { openUrl } from '@tauri-apps/plugin-opener';
import { ExternalLink, Send } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import {
  CATEGORIES,
  type Category,
  COOLDOWN_SECONDS,
  charCount,
  MAX_MESSAGE_CHARS,
  MIN_MESSAGE_CHARS,
} from './limits';

const REPO_URL = 'https://github.com/CoderDayton/lune-voice';
const ISSUES_URL = `${REPO_URL}/issues`;
const DISCUSSIONS_URL = `${REPO_URL}/discussions`;

interface HelpFeedbackDialogProps {
  open: boolean;
  onClose: () => void;
}

function ExternalLinkButton({ label, onOpen }: { label: string; onOpen: () => void }) {
  return (
    <button type="button" className="btn btn-soft justify-start gap-3" onClick={onOpen}>
      <ExternalLink size={16} strokeWidth={1.8} />
      <span className="flex-1 text-left">{label}</span>
    </button>
  );
}

function FeedbackComposer({
  category,
  onCategoryChange,
  message,
  onMessageChange,
  error,
  sent,
}: {
  category: Category;
  onCategoryChange: (next: Category) => void;
  message: string;
  onMessageChange: (next: string) => void;
  error: string | null;
  sent: boolean;
}) {
  const { t } = useLingui();
  const selectId = useId();
  const length = charCount(message.trim());
  // A native <select> rather than `DescriptiveDropdown`: that component portals
  // its menu to document.body, which paints *behind* a `showModal()` dialog
  // because the dialog sits in the browser's top layer. The native popup is
  // drawn by the OS and has no such problem.
  const labels: Record<Category, string> = {
    Bug: t`Bug report`,
    Crash: t`Crash or freeze`,
    Accuracy: t`Transcription accuracy`,
    Performance: t`Performance or speed`,
    Idea: t`Feature request`,
    Question: t`Question or help`,
    Feedback: t`General feedback`,
  };

  return (
    <>
      <div className="divider my-4 text-xs text-content-subtle">{t`or send directly`}</div>

      <label className="block" htmlFor={selectId}>
        <span className="mb-1.5 block text-xs text-content-subtle">{t`What is this about?`}</span>
        <select
          id={selectId}
          value={category}
          onChange={(e) => onCategoryChange(e.target.value as Category)}
          className="select select-bordered w-full bg-surface-elevated"
        >
          {CATEGORIES.map((id) => (
            <option key={id} value={id}>
              {labels[id]}
            </option>
          ))}
        </select>
      </label>

      <textarea
        value={message}
        onChange={(e) => onMessageChange(e.target.value)}
        maxLength={MAX_MESSAGE_CHARS}
        rows={5}
        aria-label={t`Your feedback`}
        placeholder={t`What happened, or what would you like to see?`}
        className="textarea textarea-bordered mt-3 w-full resize-none bg-surface-elevated"
      />

      <div className="mt-1 flex items-center justify-between text-xs text-content-subtle">
        <span>
          {length > 0 && length < MIN_MESSAGE_CHARS
            ? t`At least ${MIN_MESSAGE_CHARS} characters.`
            : null}
        </span>
        <span>
          {length} / {MAX_MESSAGE_CHARS}
        </span>
      </div>

      {error && (
        <div role="alert" className="alert alert-error mt-3 py-2 text-sm">
          {error}
        </div>
      )}
      {sent && !error && (
        <div role="status" className="alert alert-success mt-3 py-2 text-sm">
          {t`Thanks — your feedback was sent.`}
        </div>
      )}
    </>
  );
}

/** Probes the backend each time the dialog opens and keeps the cooldown ticking. */
function useFeedbackGate(open: boolean) {
  const [available, setAvailable] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [probeFailed, setProbeFailed] = useState(false);

  // Ask on open rather than caching: the cooldown keeps running while the
  // dialog is closed, so a stale value would show a dead button.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setProbeFailed(false);
    void (async () => {
      try {
        const [isAvailable, remaining] = await Promise.all([
          backendClient.feedbackAvailable(),
          backendClient.feedbackCooldownRemaining(),
        ]);
        if (cancelled) return;
        setAvailable(isAvailable);
        setCooldown(remaining);
      } catch {
        // A failed probe means no usable form, but say so: a composer that is
        // simply absent reads as "this build can't send feedback" — a
        // permanent state — rather than a probe that happened to fail.
        if (!cancelled) {
          setAvailable(false);
          setProbeFailed(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setInterval(() => setCooldown((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(timer);
  }, [cooldown]);

  return { available, cooldown, setCooldown, probeFailed };
}

export function HelpFeedbackDialog({ open, onClose }: HelpFeedbackDialogProps) {
  const { t } = useLingui();
  const titleId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);

  const [category, setCategory] = useState<Category>('Bug');
  const [message, setMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const { available, cooldown, setCooldown, probeFailed } = useFeedbackGate(open);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    if (!open && dlg.open) dlg.close();
    if (open) {
      setError(null);
      setSent(false);
    }
  }, [open]);

  const openExternal = useCallback(
    async (url: string) => {
      try {
        await openUrl(url);
      } catch {
        setError(t`Couldn't open your browser. Visit ${url} directly.`);
      }
    },
    [t],
  );

  const trimmed = message.trim();
  const length = charCount(trimmed);
  const blocked =
    sending || cooldown > 0 || length < MIN_MESSAGE_CHARS || length > MAX_MESSAGE_CHARS;

  // Clear the success banner as soon as the user starts a second report, so its
  // reappearance is what confirms that send rather than a leftover from the first.
  const changeMessage = useCallback((next: string) => {
    setMessage(next);
    setSent(false);
  }, []);
  const changeCategory = useCallback((next: Category) => {
    setCategory(next);
    setSent(false);
  }, []);

  async function send() {
    if (blocked) return;
    setSending(true);
    setError(null);
    try {
      await backendClient.submitFeedback(category, trimmed);
      setSent(true);
      setMessage('');
      setCooldown(COOLDOWN_SECONDS);
    } catch (e) {
      // The backend returns user-facing strings; surface them verbatim rather
      // than replacing real detail ("timed out", "not authorized") with a
      // generic failure the user can't act on.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  return (
    <dialog ref={dialogRef} className="modal" aria-labelledby={titleId} onClose={onClose}>
      <div className="modal-box max-w-lg border border-surface-border bg-surface-glass text-content">
        <h3 id={titleId} className="text-lg font-bold text-content-strong">
          {t`Help & feedback`}
        </h3>
        <p className="mt-1 text-sm text-content-muted">
          {t`Report a problem, suggest an idea, or just tell us how it's going.`}
        </p>

        <div className="mt-4 flex flex-col gap-2">
          <ExternalLinkButton
            label={t`Browse open issues`}
            onOpen={() => void openExternal(ISSUES_URL)}
          />
          <ExternalLinkButton
            label={t`Join the discussions`}
            onOpen={() => void openExternal(DISCUSSIONS_URL)}
          />
        </div>

        {available && (
          <FeedbackComposer
            category={category}
            onCategoryChange={changeCategory}
            message={message}
            onMessageChange={changeMessage}
            error={error}
            sent={sent}
          />
        )}
        {!available && probeFailed && (
          <div role="alert" className="alert alert-warning mt-3 py-2 text-sm">
            {t`Couldn't reach the feedback service. Use the links above, or reopen this dialog to retry.`}
          </div>
        )}
        {!available && error && (
          <div role="alert" className="alert alert-error mt-3 py-2 text-sm">
            {error}
          </div>
        )}

        <div className="modal-action">
          <form method="dialog">
            <button type="submit" className="btn btn-soft">
              {t`Close`}
            </button>
          </form>
          {available && (
            <button
              type="button"
              className="btn btn-primary gap-2"
              disabled={blocked}
              onClick={() => void send()}
            >
              {sending ? (
                <span className="loading loading-spinner loading-xs" />
              ) : (
                <Send size={16} strokeWidth={1.8} />
              )}
              {cooldown > 0 ? t`Wait ${cooldown}s` : t`Send`}
            </button>
          )}
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">close</button>
      </form>
    </dialog>
  );
}
