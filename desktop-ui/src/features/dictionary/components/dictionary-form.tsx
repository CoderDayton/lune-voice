import { Trans, useLingui } from '@lingui/react/macro';
import { ArrowRight, Plus } from 'lucide-react';
import { type FormEvent, useId, useRef, useState } from 'react';
import { VoiceCaptureField } from '../../../components/ui/voice-capture-field';
import { addDictionaryEntry } from '../../../store/workspace-store';
import { PriorityPicker } from './priority-picker';

/**
 * Add-a-correction composer. Collapses to a single row — "you say X → Lune
 * Voice writes Y" is itself the preview, so there's no separate confirmation
 * text; the saved-rules table below is the page's hero, not this form.
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

  const canSubmit = phrase.trim().length > 0 && replacement.trim().length > 0 && !pending;

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
    <section className="panel-surface shrink-0 p-4 sm:p-5" aria-label={t`Add a correction`}>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row sm:items-end">
        <div className="flex flex-1 flex-col gap-1.5">
          <label
            htmlFor={phraseId}
            className="text-xs font-medium uppercase tracking-wider text-content-subtle"
          >
            <Trans>You say</Trans>
          </label>
          {/* Voice capture stores the phrase exactly as the ASR hears it —
              the misheard form is precisely what this field wants. */}
          <VoiceCaptureField
            id={phraseId}
            value={phrase}
            onValueChange={setPhrase}
            placeholder={t`github`}
            inputRef={phraseRef}
            lowercase
            required
            compact
          />
        </div>

        <span
          className="hidden shrink-0 items-center justify-center pb-2.5 text-content-subtle sm:flex"
          aria-hidden="true"
        >
          <ArrowRight size={15} strokeWidth={2.2} />
        </span>

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
            className="input w-full text-sm"
            placeholder={t`GitHub`}
            autoComplete="off"
            spellCheck={false}
            required
            aria-required="true"
          />
        </div>

        <div className="flex shrink-0 flex-col gap-1.5">
          <span className="text-xs font-medium uppercase tracking-wider text-content-subtle">
            <Trans>Priority</Trans>
          </span>
          <PriorityPicker value={priority} onChange={setPriority} size="md" />
        </div>

        <button
          type="submit"
          className="btn btn-primary h-10 min-h-10 shrink-0 sm:min-w-24"
          disabled={!canSubmit}
        >
          {pending ? (
            <span className="loading loading-spinner loading-sm" />
          ) : (
            <>
              <Plus size={16} strokeWidth={2.5} />
              <Trans>Add</Trans>
            </>
          )}
        </button>
      </form>

      <p role="alert" className="mt-2 text-sm text-error empty:hidden">
        {error}
      </p>
    </section>
  );
}
