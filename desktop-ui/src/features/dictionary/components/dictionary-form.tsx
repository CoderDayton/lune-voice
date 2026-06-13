import { Trans, useLingui } from '@lingui/react/macro';
import { ArrowRight, Plus, Wand2 } from 'lucide-react';
import { type FormEvent, useId, useRef, useState } from 'react';
import { addDictionaryEntry } from '../../../store/workspace-store';
import { PriorityPicker } from './priority-picker';

/**
 * Add-a-correction composer. The two inputs read as a single mapping —
 * "you say X → Lune Voice writes Y" — joined by an arrow node, and a live preview
 * shows the misheard word being rewritten so the abstract rule feels concrete
 * before it's saved. Controlled fields drive that preview.
 */
export function QuickAddRule() {
  const { t } = useLingui();
  const phraseId = useId();
  const replacementId = useId();
  const phraseRef = useRef<HTMLInputElement>(null);

  const [phrase, setPhrase] = useState('');
  const [replacement, setReplacement] = useState('');
  const [priority, setPriority] = useState(5);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = phrase.trim().length > 0 && replacement.trim().length > 0;
  const canSubmit = ready && !pending;

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);
    const ok = await addDictionaryEntry(phrase, replacement, priority);
    setPending(false);
    if (ok) {
      setPhrase('');
      setReplacement('');
      setPriority(5);
      phraseRef.current?.focus();
    } else {
      setError(t`Couldn't add that rule. Try again.`);
    }
  }

  return (
    <section className="panel-surface p-5 sm:p-6" aria-labelledby="vocab-add-heading">
      {/* Header */}
      <div className="flex items-start gap-3">
        <span className="token-icon grid size-10 shrink-0 place-items-center rounded-field">
          <Wand2 size={19} strokeWidth={1.6} />
        </span>
        <div className="min-w-0">
          <h2
            id="vocab-add-heading"
            className="text-lg font-semibold tracking-tight text-content-strong"
          >
            <Trans>Add a correction</Trans>
          </h2>
          <p className="text-sm text-content-subtle">
            <Trans>Teach Lune Voice the right spelling for a word it mishears.</Trans>
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-4">
        {/* Mapping: you say → Lune Voice writes */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:gap-3">
          <div className="flex flex-1 flex-col gap-1.5">
            <label
              htmlFor={phraseId}
              className="text-xs font-medium uppercase tracking-wider text-content-subtle"
            >
              <Trans>You say</Trans>
            </label>
            <input
              id={phraseId}
              ref={phraseRef}
              type="text"
              value={phrase}
              onChange={(e) => setPhrase(e.target.value)}
              className="input input-lg w-full text-base"
              placeholder={t`github`}
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              required
              aria-required="true"
            />
          </div>

          <div
            className="flex shrink-0 items-center justify-center self-center pb-0 sm:mb-1.5 sm:self-end"
            aria-hidden="true"
          >
            <span className="grid size-8 place-items-center rounded-full border border-surface-border bg-surface-panel text-content-dim">
              <ArrowRight size={15} strokeWidth={2.2} className="rotate-90 sm:rotate-0" />
            </span>
          </div>

          <div className="flex flex-1 flex-col gap-1.5">
            <label
              htmlFor={replacementId}
              className="text-xs font-medium uppercase tracking-wider text-content-subtle"
            >
              <Trans>Lune Voice writes</Trans>
            </label>
            <input
              id={replacementId}
              type="text"
              value={replacement}
              onChange={(e) => setReplacement(e.target.value)}
              className="input input-lg w-full text-base"
              placeholder={t`GitHub`}
              autoComplete="off"
              spellCheck={false}
              required
              aria-required="true"
            />
          </div>
        </div>

        {/* Live preview — the misheard word rewritten to the correction */}
        <div
          aria-live="polite"
          className="rounded-field border border-surface-border bg-surface-inset/60 px-4 py-2.5 text-sm"
        >
          {ready ? (
            <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-content-muted">
              <span className="text-xs uppercase tracking-wider text-content-dim">
                <Trans>Preview</Trans>
              </span>
              <span className="text-content-subtle line-through decoration-content-dim/60">
                {phrase}
              </span>
              <ArrowRight size={13} strokeWidth={2} className="text-content-dim" />
              <span className="font-medium text-primary">{replacement}</span>
            </span>
          ) : (
            <span className="text-content-dim">
              <Trans>Fill both fields to see the correction.</Trans>
            </span>
          )}
        </div>

        {/* Priority + submit */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium uppercase tracking-wider text-content-subtle">
              <Trans>Priority</Trans>
            </span>
            <PriorityPicker value={priority} onChange={setPriority} size="sm" />
          </div>
          <button type="submit" className="btn btn-primary sm:min-w-36" disabled={!canSubmit}>
            {pending ? (
              <span className="loading loading-spinner loading-sm" />
            ) : (
              <>
                <Plus size={16} strokeWidth={2.5} />
                <Trans>Add correction</Trans>
              </>
            )}
          </button>
        </div>

        <p role="alert" className="text-sm text-error empty:hidden">
          {error}
        </p>
      </form>
    </section>
  );
}
