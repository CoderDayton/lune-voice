// SPDX-License-Identifier: GPL-3.0-or-later

import { plural } from '@lingui/core/macro';
import { Trans, useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { CheckCheck, Mic, Pencil, Plus, ScrollText, Trash2, X } from 'lucide-react';
import { type FormEvent, type Ref, useEffect, useId, useMemo, useRef, useState } from 'react';
import { ConfirmDialog } from '../../components/feedback/confirm-dialog';
import { Fold } from '../../components/ui/fold';
import {
  LIST_HEAD_BAR,
  LIST_ROW,
  LIST_ROW_ACTION,
  ListCard,
  ListEmpty,
  ListFooter,
  ListScroll,
  ListSearchRail,
  useListFilterHotkey,
} from '../../components/ui/list-card';
import { VoiceCaptureField } from '../../components/ui/voice-capture-field';
import type { SnippetItemDto } from '../../lib/backend/types';
import { removeSnippet, saveSnippet, workspaceStore } from '../../store/workspace-store';
import { PLACEHOLDERS, previewBody } from './snippet-preview';

/**
 * Snippets page. The composer (create or edit one voice-triggered template)
 * sits above the saved list, and the list takes the rest of the pane — the
 * same shape as Sessions and Vocabulary, sharing their card chrome via
 * `components/ui/list-card`.
 *
 * The composer teaches by showing: a "say it" hint spells out the exact spoken
 * forms, and the live preview renders the placeholders with real values.
 */
export function SnippetsPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack min-h-0 flex-1">
        <div className="panel-surface shrink-0 space-y-4 p-4 sm:p-5">
          <div className="skeleton-shimmer h-12 w-full rounded-field" />
          <div className="skeleton-shimmer h-24 w-full rounded-field" />
        </div>
        <div className="min-h-0 flex-1 divide-y divide-surface-border overflow-hidden">
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

/**
 * Composer state, shared by the form and the list: the list starts an edit,
 * the form finishes it. Kept in one hook so neither owns the other.
 */
function useSnippetComposer({ onSaved }: { onSaved: () => void }) {
  const { t } = useLingui();
  const triggerRef = useRef<HTMLInputElement>(null);
  const bodyRef = useRef<HTMLTextAreaElement>(null);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [trigger, setTrigger] = useState('');
  const [body, setBody] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reset() {
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

  const canSubmit = trigger.trim().length > 0 && body.trim().length > 0 && !pending;

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);
    const ok = await saveSnippet(editingId, trigger, body);
    setPending(false);
    if (ok) {
      reset();
      onSaved();
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

  return {
    triggerRef,
    bodyRef,
    trigger,
    setTrigger,
    body,
    setBody,
    editing: editingId !== null,
    pending,
    error,
    canSubmit,
    reset,
    startEdit,
    submit,
    insertPlaceholder,
  };
}

type Composer = ReturnType<typeof useSnippetComposer>;

function SnippetsContent() {
  const { t } = useLingui();
  // The composer is five times the height of Vocabulary's one-line one, and
  // snippets are written far less often than they are browsed — so the list is
  // the resting state and the form folds away until asked for. Starting an edit
  // from the list opens it too, which is why `editing` forces it open.
  const [open, setOpen] = useState(false);
  const composer = useSnippetComposer({ onSaved: () => collapse() });
  // With no snippets saved, the empty state below carries the call to action —
  // a second identical one above it would just be noise.
  const hasSnippets = useStore(workspaceStore, (s) => s.snippets.length > 0);

  // Collapsing unmounts whatever the user was on (Save, Cancel), which would
  // otherwise drop focus onto <body> and lose a keyboard user's place.
  const barRef = useRef<HTMLButtonElement>(null);
  const restoreFocus = useRef(false);
  const collapsed = !open && !composer.editing;
  useEffect(() => {
    if (collapsed && restoreFocus.current) {
      restoreFocus.current = false;
      barRef.current?.focus();
    }
  }, [collapsed]);

  function collapse() {
    restoreFocus.current = true;
    setOpen(false);
  }

  function close() {
    composer.reset();
    collapse();
  }

  return (
    <div className="page-stack min-h-0 flex-1">
      {/* The bar and the composer occupy the same slot, so the composer grows
          out of the row that was clicked instead of appearing above it. The
          bar is only hidden once the unfold has started, which is why it keys
          off `open` rather than waiting for `collapsed`. */}
      <Fold open={!collapsed} className="shrink-0">
        <SnippetComposer composer={composer} onClose={close} />
      </Fold>
      {collapsed && hasSnippets && <NewSnippetBar ref={barRef} onClick={() => setOpen(true)} />}
      <section aria-label={t`Saved snippets`} className="flex min-h-0 flex-1 flex-col">
        <SnippetList onEdit={composer.startEdit} onNew={() => setOpen(true)} />
      </section>
    </div>
  );
}

/** Collapsed composer: one row that opens the form, echoing its Add button. */
function NewSnippetBar({ ref, onClick }: { ref?: Ref<HTMLButtonElement>; onClick: () => void }) {
  return (
    <button
      ref={ref}
      type="button"
      onClick={onClick}
      className="panel-surface flex shrink-0 items-center gap-2.5 px-4 py-3.5 text-sm font-medium text-content-muted transition-colors hover:border-surface-border-strong hover:bg-surface-hover hover:text-content sm:px-5"
    >
      <Plus size={16} strokeWidth={2.5} className="text-primary" />
      <Trans>New snippet</Trans>
    </button>
  );
}

/**
 * Create/edit form. No panel header: the page header above already names the
 * feature, so a second title would spend the height the list needs. Editing
 * announces itself with an inline banner instead, which also says *which*
 * snippet is open — something a static "Edit snippet" title never did.
 */
function SnippetComposer({ composer: c, onClose }: { composer: Composer; onClose: () => void }) {
  const { t } = useLingui();
  const triggerId = useId();
  const bodyId = useId();
  const spokenTrigger = c.trigger.trim().toLowerCase();
  const { triggerRef } = c;

  // Opening the form should land the caret in it, however it was opened.
  useEffect(() => {
    triggerRef.current?.focus();
  }, [triggerRef]);

  return (
    <section
      className="panel-surface shrink-0 p-4 sm:p-5"
      aria-label={c.editing ? t`Edit snippet` : t`Create a snippet`}
    >
      <div
        className={`mb-3 flex items-center justify-between gap-3 ${
          c.editing ? 'rounded-field border border-primary/30 bg-primary/10 px-3 py-2' : ''
        }`}
      >
        <span
          className={`min-w-0 truncate text-sm ${
            c.editing ? 'text-primary' : 'font-medium text-content'
          }`}
        >
          {c.editing ? <Trans>Editing “{spokenTrigger}”</Trans> : <Trans>New snippet</Trans>}
        </span>
        <button type="button" onClick={onClose} className="btn btn-soft btn-sm shrink-0 gap-1.5">
          <X size={14} />
          <Trans>Cancel</Trans>
        </button>
      </div>

      <form onSubmit={c.submit} className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <label
            htmlFor={triggerId}
            className="text-xs font-medium uppercase tracking-wider text-content-subtle"
          >
            <Trans>Trigger phrase</Trans>
          </label>
          <VoiceCaptureField
            id={triggerId}
            value={c.trigger}
            onValueChange={c.setTrigger}
            placeholder={t`meeting header`}
            inputRef={c.triggerRef}
            required
            compact
            hint={
              spokenTrigger ? (
                <Trans>
                  Say <span className="font-medium text-content-muted">“{spokenTrigger}”</span> as
                  the whole utterance — or{' '}
                  <span className="font-medium text-content-muted">“insert {spokenTrigger}”</span> —
                  to type this snippet.
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
            {/* Tokens, not buttons: they read as the literal text they drop
                in, so mono and bordered rather than a row of filled chips. */}
            <fieldset className="flex flex-wrap gap-1.5" aria-label={t`Insert placeholder`}>
              {PLACEHOLDERS.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => c.insertPlaceholder(name)}
                  className="rounded-selector border border-surface-border bg-surface-field px-2 py-1 font-mono text-xs text-content-muted transition-colors hover:border-primary/50 hover:bg-primary/15 hover:text-primary"
                  title={t`Insert at cursor`}
                >
                  {`{${name}}`}
                </button>
              ))}
            </fieldset>
          </div>
          <textarea
            id={bodyId}
            ref={c.bodyRef}
            value={c.body}
            onChange={(e) => c.setBody(e.target.value)}
            className="textarea w-full font-mono text-sm leading-6"
            rows={4}
            placeholder={t`## Meeting — {date}\n\nAttendees:\n- `}
            spellCheck={false}
            required
            aria-required="true"
          />
        </div>

        {/* Live preview — placeholders rendered with real values. Only while
            there is something to preview; an empty explainer box would cost
            the list a row for nothing. */}
        {c.body.trim() && (
          <div
            aria-live="polite"
            className="rounded-2xl border border-surface-border bg-surface-inset/60 px-3 py-2 text-sm"
          >
            <span className="text-xs uppercase tracking-wider text-content-subtle">
              <Trans>Types as</Trans>
            </span>
            <pre className="mt-1 max-h-28 overflow-y-auto whitespace-pre-wrap break-words font-sans text-content-muted">
              {previewBody(c.body, t`(clipboard contents)`)}
            </pre>
          </div>
        )}

        <div className="flex items-center justify-end gap-3">
          <p role="alert" className="min-w-0 flex-1 text-sm text-error empty:hidden">
            {c.error}
          </p>
          <button
            type="submit"
            className="btn btn-primary h-10 min-h-10 shrink-0 sm:min-w-36"
            disabled={!c.canSubmit}
          >
            {c.pending ? (
              <span className="loading loading-spinner loading-sm" />
            ) : c.editing ? (
              <Trans>Save changes</Trans>
            ) : (
              <>
                <Plus size={16} strokeWidth={2.5} />
                <Trans>Add snippet</Trans>
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  );
}

/** Case-insensitive match of a query against a snippet's trigger or body. */
function snippetMatches(snippet: SnippetItemDto, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return snippet.triggerPhrase.toLowerCase().includes(q) || snippet.body.toLowerCase().includes(q);
}

/**
 * Saved snippets. Same card, rail, column labels, rows, and footer as Sessions
 * and Vocabulary — a `<ul>` rather than a table because a snippet body is a
 * block of text, not a cell.
 */
function SnippetList({
  onEdit,
  onNew,
}: {
  onEdit: (snippet: SnippetItemDto) => void;
  onNew: () => void;
}) {
  const { t } = useLingui();
  const snippets = useStore(workspaceStore, (s) => s.snippets);
  const [query, setQuery] = useState('');
  const queryRef = useRef<HTMLInputElement>(null);

  useListFilterHotkey(queryRef);

  const filtered = useMemo(
    () => snippets.filter((snippet) => snippetMatches(snippet, query)),
    [snippets, query],
  );

  if (snippets.length === 0) return <SnippetsEmptyState onNew={onNew} />;

  return (
    <ListCard>
      <ListSearchRail
        inputRef={queryRef}
        value={query}
        onValueChange={setQuery}
        placeholder={t`Filter snippets…`}
        label={t`Filter snippets`}
        count={
          query.trim()
            ? t`${filtered.length} / ${snippets.length}`
            : plural(snippets.length, { one: '# snippet', other: '# snippets' })
        }
      />

      <div className={LIST_HEAD_BAR}>
        {/* The rows lead with a mic glyph, so the label has to clear the same
            width or the column reads as two columns that missed each other. */}
        <span className="flex w-40 shrink-0 items-center gap-1.5 sm:w-56">
          <Mic size={13} strokeWidth={2} className="invisible shrink-0" aria-hidden="true" />
          <Trans>Trigger</Trans>
        </span>
        <span className="min-w-0 flex-1">
          <Trans>Snippet text</Trans>
        </span>
      </div>

      <ListScroll>
        {filtered.length === 0 ? (
          <ListEmpty>
            <p role="status" className="text-sm text-content-muted">
              <Trans>No snippets match “{query}”.</Trans>
            </p>
          </ListEmpty>
        ) : (
          <ul>
            {filtered.map((snippet) => (
              <li
                key={snippet.id}
                className={`${LIST_ROW} group flex items-start gap-4 px-3 py-3 sm:px-4`}
              >
                <p className="flex w-40 shrink-0 items-center gap-1.5 text-sm font-medium text-content sm:w-56">
                  <Mic
                    size={13}
                    strokeWidth={2}
                    className="shrink-0 text-content-subtle"
                    aria-hidden="true"
                  />
                  <span dir="auto" className="truncate">
                    {snippet.triggerPhrase}
                  </span>
                </p>
                <p
                  dir="auto"
                  className="line-clamp-2 min-w-0 flex-1 whitespace-pre-line font-mono text-sm text-content-muted"
                >
                  {snippet.body}
                </p>
                <div className="flex shrink-0 items-center gap-1">
                  <button
                    type="button"
                    onClick={() => onEdit(snippet)}
                    className={`${LIST_ROW_ACTION} btn-soft`}
                    aria-label={t`Edit ${snippet.triggerPhrase}`}
                  >
                    <Pencil size={14} />
                  </button>
                  <ConfirmDialog
                    title={t`Delete snippet?`}
                    description={t`“${snippet.triggerPhrase}” will stop expanding. The template text is not recoverable.`}
                    triggerLabel={t`Delete ${snippet.triggerPhrase}`}
                    triggerIcon={<Trash2 size={14} />}
                    triggerClassName={`${LIST_ROW_ACTION} btn-soft btn-error`}
                    onConfirm={() => removeSnippet(snippet.id)}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}
      </ListScroll>

      <ListFooter>
        <span className="flex items-center gap-2 text-sm text-content-subtle">
          <CheckCheck size={14} strokeWidth={1.9} className="text-success" aria-hidden="true" />
          <Trans>Say a trigger as your whole utterance to expand it</Trans>
        </span>
      </ListFooter>
    </ListCard>
  );
}

/** First-run state, filling the pane the way Vocabulary's does. With the
 *  composer folded away, this carries the only call to action. */
function SnippetsEmptyState({ onNew }: { onNew: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 py-12 text-center">
      <div className="token-icon mb-4 size-14 rounded-full">
        <ScrollText className="size-7" strokeWidth={1.4} />
      </div>
      <h3 className="text-lg font-semibold tracking-tight text-content-strong">
        <Trans>No snippets yet</Trans>
      </h3>
      <p className="mt-1.5 max-w-sm text-sm text-content-subtle">
        <Trans>
          Save your signature, a meeting header, or anything you type often — then speak its trigger
          to drop it into any app.
        </Trans>
      </p>
      <button type="button" onClick={onNew} className="btn btn-primary mt-5 gap-1.5">
        <Plus size={16} strokeWidth={2.5} />
        <Trans>New snippet</Trans>
      </button>
    </div>
  );
}
