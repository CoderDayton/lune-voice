import { Trans, useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Mic, Pencil, Plus, ScrollText, X } from 'lucide-react';
import { type FormEvent, useId, useRef, useState } from 'react';
import { ConfirmDialog } from '../../components/feedback/confirm-dialog';
import { PanelHeader } from '../../components/ui/panel-header';
import { VoiceCaptureField } from '../../components/ui/voice-capture-field';
import type { SnippetItemDto } from '../../lib/backend/types';
import { removeSnippet, saveSnippet, workspaceStore } from '../../store/workspace-store';

/**
 * Placeholders the backend substitutes at expansion time (see
 * `lune-dict::snippets::render_body`). The preview below mirrors the same
 * semantics: known tokens are replaced, anything else in braces passes
 * through verbatim.
 */
const PLACEHOLDERS = ['date', 'time', 'datetime', 'clipboard'] as const;

function previewBody(body: string, clipboardLabel: string): string {
  const now = new Date();
  const date = now.toLocaleDateString(undefined, {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  const time = now.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  return body.replace(/\{([^{}]*)\}/g, (raw, token: string) => {
    switch (token.toLowerCase()) {
      case 'date':
        return date;
      case 'time':
        return time;
      case 'datetime':
        return `${date} at ${time}`;
      case 'clipboard':
        return clipboardLabel;
      default:
        return raw;
    }
  });
}

/**
 * Snippets page. The composer (create or edit one voice-triggered template)
 * sits above the saved list, mirroring the Vocabulary page's layout. The
 * composer teaches by showing: a "say it" hint spells out the exact spoken
 * forms, and the live preview renders the placeholders with real values.
 */
export function SnippetsPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack">
        <div className="panel-surface space-y-4 p-5 sm:p-6">
          <div className="flex items-center gap-3">
            <div className="skeleton-shimmer size-10 rounded-field" />
            <div className="space-y-2">
              <div className="skeleton-shimmer h-4 w-40" />
              <div className="skeleton-shimmer h-3 w-56" />
            </div>
          </div>
          <div className="skeleton-shimmer h-12 w-full rounded-field" />
          <div className="skeleton-shimmer h-24 w-full rounded-field" />
        </div>
        <div className="panel-surface divide-y divide-surface-border overflow-hidden">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-4">
              <div className="skeleton-shimmer h-4 w-32" />
              <div className="skeleton-shimmer h-4 flex-1" />
              <div className="skeleton-shimmer h-6 w-16 rounded-field" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return <SnippetsContent />;
}

function SnippetsContent() {
  const { t } = useLingui();
  const snippets = useStore(workspaceStore, (s) => s.snippets);
  const triggerId = useId();
  const bodyId = useId();
  const triggerRef = useRef<HTMLInputElement>(null);
  const bodyRef = useRef<HTMLTextAreaElement>(null);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [trigger, setTrigger] = useState('');
  const [body, setBody] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = trigger.trim().length > 0 && body.trim().length > 0;
  const canSubmit = ready && !pending;
  const editing = editingId !== null;

  function resetComposer() {
    setEditingId(null);
    setTrigger('');
    setBody('');
    setError(null);
  }

  function startEdit(snippet: SnippetItemDto) {
    setEditingId(snippet.id);
    setTrigger(snippet.triggerPhrase);
    setBody(snippet.body);
    setError(null);
    triggerRef.current?.focus();
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);
    const ok = await saveSnippet(editingId, trigger, body);
    setPending(false);
    if (ok) {
      resetComposer();
      triggerRef.current?.focus();
    } else {
      setError(t`Couldn't save that snippet. Try again.`);
    }
  }

  /** Insert `{name}` at the caret, keeping the caret just after the token. */
  function insertPlaceholder(name: string) {
    const token = `{${name}}`;
    const el = bodyRef.current;
    if (!el) {
      setBody((b) => b + token);
      return;
    }
    const start = el.selectionStart ?? body.length;
    const end = el.selectionEnd ?? start;
    setBody(body.slice(0, start) + token + body.slice(end));
    requestAnimationFrame(() => {
      el.focus();
      const pos = start + token.length;
      el.setSelectionRange(pos, pos);
    });
  }

  const spokenTrigger = trigger.trim().toLowerCase();

  return (
    <div className="page-stack">
      <section
        className="panel-surface p-5 sm:p-6"
        aria-label={editing ? t`Edit snippet` : t`Create a snippet`}
      >
        <PanelHeader
          icon={ScrollText}
          title={editing ? t`Edit snippet` : t`Create a snippet`}
          description={t`A block of text Lune Voice types for you when you speak its trigger phrase.`}
          actions={
            editing ? (
              <button type="button" onClick={resetComposer} className="btn btn-soft btn-sm gap-1.5">
                <X size={14} />
                <Trans>Cancel edit</Trans>
              </button>
            ) : undefined
          }
        />

        <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label
              htmlFor={triggerId}
              className="text-xs font-medium uppercase tracking-wider text-content-subtle"
            >
              <Trans>Trigger phrase</Trans>
            </label>
            <VoiceCaptureField
              id={triggerId}
              value={trigger}
              onValueChange={setTrigger}
              placeholder={t`meeting header`}
              inputRef={triggerRef}
              required
              hint={
                spokenTrigger ? (
                  <Trans>
                    Say <span className="font-medium text-content-muted">“{spokenTrigger}”</span> as
                    the whole utterance — or{' '}
                    <span className="font-medium text-content-muted">“insert {spokenTrigger}”</span>{' '}
                    — to type this snippet.
                  </Trans>
                ) : (
                  <Trans>
                    Type a phrase, or tap the mic to capture it exactly as Lune Voice hears it.
                  </Trans>
                )
              }
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex flex-wrap items-end justify-between gap-2">
              <label
                htmlFor={bodyId}
                className="text-xs font-medium uppercase tracking-wider text-content-subtle"
              >
                <Trans>Snippet text</Trans>
              </label>
              <fieldset className="flex flex-wrap gap-1.5" aria-label={t`Insert placeholder`}>
                {PLACEHOLDERS.map((name) => (
                  <button
                    key={name}
                    type="button"
                    onClick={() => insertPlaceholder(name)}
                    className="btn btn-soft btn-xs font-mono normal-case"
                    title={t`Insert at cursor`}
                  >
                    {`{${name}}`}
                  </button>
                ))}
              </fieldset>
            </div>
            <textarea
              id={bodyId}
              ref={bodyRef}
              value={body}
              onChange={(e) => setBody(e.target.value)}
              className="textarea w-full font-mono text-sm leading-6"
              rows={5}
              placeholder={t`## Meeting — {date}\n\nAttendees:\n- `}
              spellCheck={false}
              required
              aria-required="true"
            />
          </div>

          {/* Live preview — placeholders rendered with real values */}
          <div
            aria-live="polite"
            className="rounded-field border border-surface-border bg-surface-inset/60 px-4 py-2.5 text-sm"
          >
            {body.trim() ? (
              <>
                <span className="text-xs uppercase tracking-wider text-content-subtle">
                  <Trans>Types as</Trans>
                </span>
                <pre className="mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap break-words font-sans text-content-muted">
                  {previewBody(body, t`(clipboard contents)`)}
                </pre>
              </>
            ) : (
              <span className="text-content-subtle">
                <Trans>The snippet text appears here with placeholders filled in.</Trans>
              </span>
            )}
          </div>

          <div className="flex items-center justify-end">
            <button type="submit" className="btn btn-primary sm:min-w-36" disabled={!canSubmit}>
              {pending ? (
                <span className="loading loading-spinner loading-sm" />
              ) : editing ? (
                <Trans>Save changes</Trans>
              ) : (
                <>
                  <Plus size={16} strokeWidth={2.5} />
                  <Trans>Add snippet</Trans>
                </>
              )}
            </button>
          </div>

          <p role="alert" className="text-sm text-error empty:hidden">
            {error}
          </p>
        </form>
      </section>

      <section aria-label={t`Saved snippets`}>
        {snippets.length === 0 ? (
          <div className="panel-surface flex flex-col items-center gap-2 px-6 py-12 text-center">
            <ScrollText size={28} strokeWidth={1.5} className="text-content-subtle" />
            <p className="text-sm text-content-muted">
              <Trans>No snippets yet.</Trans>
            </p>
            <p className="max-w-md text-sm text-content-subtle">
              <Trans>
                Save your signature, a meeting header, or anything you type often — then speak its
                trigger to drop it into any app.
              </Trans>
            </p>
          </div>
        ) : (
          <ul className="panel-surface divide-y divide-surface-border overflow-hidden">
            {snippets.map((snippet) => (
              <li key={snippet.id} className="flex items-start gap-4 px-5 py-4">
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-1.5 font-medium text-content">
                    <Mic
                      size={13}
                      strokeWidth={2}
                      className="shrink-0 text-content-subtle"
                      aria-hidden="true"
                    />
                    {snippet.triggerPhrase}
                  </p>
                  <p className="mt-1 line-clamp-2 whitespace-pre-line text-sm text-content-muted">
                    {snippet.body}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={() => startEdit(snippet)}
                    className="btn btn-soft btn-sm gap-1.5"
                  >
                    <Pencil size={14} />
                    <Trans>Edit</Trans>
                  </button>
                  <ConfirmDialog
                    title={t`Delete snippet?`}
                    description={t`“${snippet.triggerPhrase}” will stop expanding. The template text is not recoverable.`}
                    triggerLabel={t`Delete`}
                    onConfirm={() => removeSnippet(snippet.id)}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
