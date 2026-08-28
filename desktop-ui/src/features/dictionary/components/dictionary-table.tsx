import { Trans, useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { ArrowRight, Check, CheckCheck, Pencil, Sparkles, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ConfirmDialog } from '../../../components/feedback';
import {
  LIST_HEAD_CELL,
  LIST_ROW,
  LIST_ROW_ACTION,
  ListCard,
  ListEmpty,
  ListFooter,
  ListScroll,
  ListSearchRail,
  useListFilterHotkey,
} from '../../../components/ui/list-card';
import {
  addDictionaryEntry,
  removeDictionaryEntry,
  updateDictionaryEntry,
  workspaceStore,
} from '../../../store/workspace-store';
import { type DictionaryEntry, entryMatchesQuery, VOCAB_PRESETS } from '../data/dictionary-data';
import { PriorityBadge, PriorityPicker } from './priority-picker';

function EditRow({ entry, onDone }: { entry: DictionaryEntry; onDone: () => void }) {
  const { t } = useLingui();
  const [phrase, setPhrase] = useState(entry.phrase);
  const [replacement, setReplacement] = useState(entry.replacement);
  const [priority, setPriority] = useState(entry.priority);
  const [saving, setSaving] = useState(false);
  const phraseRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    phraseRef.current?.focus();
    phraseRef.current?.select();
  }, []);

  const dirty =
    phrase !== entry.phrase || replacement !== entry.replacement || priority !== entry.priority;
  const valid = phrase.trim().length > 0 && replacement.trim().length > 0;

  const save = useCallback(async () => {
    if (!dirty || !valid) return;
    setSaving(true);
    const ok = await updateDictionaryEntry(entry.id, phrase, replacement, priority);
    setSaving(false);
    if (ok) onDone();
  }, [dirty, valid, phrase, replacement, priority, entry.id, onDone]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        void save();
      } else if (e.key === 'Escape') {
        onDone();
      }
    },
    [save, onDone],
  );

  return (
    <tr
      className="border-b border-surface-border/60 bg-surface-nav-active align-top"
      onKeyDown={handleKeyDown}
    >
      <td className="px-3 py-3">
        <input
          ref={phraseRef}
          type="text"
          className="input input-sm w-full"
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          placeholder={t`Phrase`}
          aria-label={t`Spoken phrase`}
        />
      </td>
      <td colSpan={2} className="hidden px-2 py-3 sm:table-cell">
        <div className="flex items-center gap-3">
          <input
            type="text"
            className="input input-sm flex-1"
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            placeholder={t`Replacement`}
            aria-label={t`Replacement`}
          />
          <PriorityPicker value={priority} onChange={setPriority} />
        </div>
      </td>
      <td className="py-3 pr-3">
        <div className="mb-2 space-y-2 sm:hidden">
          <input
            type="text"
            className="input input-sm w-full"
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            placeholder={t`Replacement`}
            aria-label={t`Replacement`}
          />
          <PriorityPicker value={priority} onChange={setPriority} />
        </div>
        <div className="flex items-center justify-end gap-1">
          <button
            type="button"
            className="btn btn-soft btn-sm btn-square"
            onClick={onDone}
            disabled={saving}
            aria-label={t`Cancel`}
          >
            <X size={15} />
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm btn-square"
            onClick={() => void save()}
            disabled={saving || !dirty || !valid}
            aria-label={t`Save`}
          >
            {saving ? <span className="loading loading-spinner loading-xs" /> : <Check size={15} />}
          </button>
        </div>
      </td>
    </tr>
  );
}

function ViewRow({ entry, onEdit }: { entry: DictionaryEntry; onEdit: () => void }) {
  const { t } = useLingui();

  return (
    <tr className={`${LIST_ROW} group`}>
      <td className="px-3 py-3">
        <div className="text-sm font-medium text-content">{entry.phrase}</div>
        <div className="mt-0.5 flex items-center gap-1 text-xs text-content-subtle sm:hidden">
          <ArrowRight size={11} strokeWidth={2} /> {entry.replacement}
        </div>
      </td>
      <td className="hidden px-2 py-3 font-mono text-sm text-content-muted sm:table-cell">
        {entry.replacement}
      </td>
      <td className="hidden px-2 py-3 sm:table-cell">
        <PriorityBadge value={entry.priority} />
      </td>
      <td className="py-3 pr-3 text-right">
        <div className="flex items-center justify-end gap-1">
          <span className="sm:hidden">
            <PriorityBadge value={entry.priority} />
          </span>
          <button
            type="button"
            className={`${LIST_ROW_ACTION} btn-soft`}
            onClick={onEdit}
            aria-label={t`Edit ${entry.phrase}`}
          >
            <Pencil size={14} />
          </button>
          <ConfirmDialog
            title={t`Delete this word fix?`}
            description={t`Remove the saved correction for "${entry.phrase}" → "${entry.replacement}".`}
            triggerLabel={t`Delete ${entry.phrase}`}
            triggerIcon={<Trash2 size={14} />}
            triggerClassName={`${LIST_ROW_ACTION} btn-soft btn-error`}
            onConfirm={() => void removeDictionaryEntry(entry.id)}
          />
        </div>
      </td>
    </tr>
  );
}

/** First-run state: teach by example. One tap on a preset seeds a real rule. */
function EmptyState() {
  const [seeding, setSeeding] = useState<string | null>(null);

  return (
    <div className="panel-surface flex min-h-0 flex-1 flex-col items-center justify-center px-6 py-12 text-center sm:py-16">
      <div className="token-icon mb-4 size-14 rounded-full">
        <Sparkles className="size-7" strokeWidth={1.4} />
      </div>
      <h3 className="text-lg font-semibold tracking-tight text-content-strong">
        <Trans>Teach Lune Voice your words</Trans>
      </h3>
      <p className="mt-1.5 max-w-sm text-sm text-content-subtle">
        <Trans>
          Add a fix and Lune Voice applies it every time you dictate. Try a starter below or add
          your own above.
        </Trans>
      </p>
      <ul className="mt-5 flex max-w-md flex-wrap justify-center gap-2">
        {VOCAB_PRESETS.map((preset) => {
          const key = `${preset.phrase}→${preset.replacement}`;
          return (
            <li key={key}>
              <button
                type="button"
                disabled={seeding !== null}
                onClick={async () => {
                  setSeeding(key);
                  await addDictionaryEntry(preset.phrase, preset.replacement, preset.priority);
                  setSeeding(null);
                }}
                className="inline-flex items-center gap-1.5 rounded-full border border-surface-border bg-surface-panel px-3 py-1.5 text-sm text-content-muted transition-colors hover:border-surface-border-strong hover:text-content disabled:opacity-50"
              >
                {seeding === key ? (
                  <span className="loading loading-spinner loading-xs" />
                ) : (
                  <span className="text-content">{preset.phrase}</span>
                )}
                <ArrowRight size={12} strokeWidth={2} className="text-content-subtle" />
                <span className="font-medium text-content">{preset.replacement}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function VocabularyList() {
  const { t } = useLingui();
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const queryRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(
    () => entries.filter((entry) => entryMatchesQuery(entry, query)),
    [entries, query],
  );

  useListFilterHotkey(queryRef);

  if (entries.length === 0) return <EmptyState />;

  return (
    // Same card, rail, and table chrome as Sessions (`components/ui/list-card`).
    // The filter belongs *to* the table, so it stays put when a query matches
    // nothing instead of vanishing along with the results.
    <ListCard>
      <ListSearchRail
        inputRef={queryRef}
        value={query}
        onValueChange={setQuery}
        placeholder={t`Filter rules…`}
        label={t`Filter rules`}
        count={
          query.trim() ? t`${filtered.length} / ${entries.length}` : t`${entries.length} rules`
        }
      />

      <ListScroll>
        {filtered.length === 0 ? (
          <ListEmpty>
            <p role="status" className="text-sm text-content-muted">
              <Trans>No rules match “{query}”.</Trans>
            </p>
          </ListEmpty>
        ) : (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className={`${LIST_HEAD_CELL} px-3`}>
                  <Trans>Phrase</Trans>
                </th>
                <th className={`${LIST_HEAD_CELL} hidden px-2 sm:table-cell`}>
                  <Trans>Replacement</Trans>
                </th>
                <th className={`${LIST_HEAD_CELL} hidden w-28 px-2 sm:table-cell`}>
                  <Trans>Priority</Trans>
                </th>
                <th className={`${LIST_HEAD_CELL} w-20 pr-3 sm:w-24`}>
                  <span className="sr-only">
                    <Trans>Action</Trans>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry) =>
                editingId === entry.id ? (
                  <EditRow key={entry.id} entry={entry} onDone={() => setEditingId(null)} />
                ) : (
                  <ViewRow key={entry.id} entry={entry} onEdit={() => setEditingId(entry.id)} />
                ),
              )}
            </tbody>
          </table>
        )}
      </ListScroll>

      <ListFooter>
        <span className="flex items-center gap-2 text-sm text-content-subtle">
          <CheckCheck size={14} strokeWidth={1.9} className="text-success" aria-hidden="true" />
          <Trans>Applied on every dictation · highest priority first</Trans>
        </span>
      </ListFooter>
    </ListCard>
  );
}
