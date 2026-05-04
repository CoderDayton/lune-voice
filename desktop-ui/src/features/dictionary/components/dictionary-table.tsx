import { useState, useRef, useEffect, useCallback } from "react";
import { useStore } from "@tanstack/react-store";
import { ConfirmDialog } from "../../../components/feedback";
import {
  removeDictionaryEntry,
  updateDictionaryEntry,
  workspaceStore,
} from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";
import { BookOpen, Check, X, Pencil } from "lucide-react";
import type { DictionaryEntry } from "../data/dictionary-data";
import { PriorityBadge, PriorityPicker } from "./priority-picker";

function EditRow({
  entry,
  onDone,
}: {
  entry: DictionaryEntry;
  onDone: () => void;
}) {
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
    phrase !== entry.phrase ||
    replacement !== entry.replacement ||
    priority !== entry.priority;

  const save = useCallback(async () => {
    if (!dirty || !phrase.trim() || !replacement.trim()) return;
    setSaving(true);
    const ok = await updateDictionaryEntry(
      entry.id,
      phrase,
      replacement,
      priority,
    );
    setSaving(false);
    if (ok) onDone();
  }, [dirty, phrase, replacement, priority, entry.id, onDone]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        void save();
      } else if (e.key === "Escape") {
        onDone();
      }
    },
    [save, onDone],
  );

  return (
    <tr className="bg-base-200/40" onKeyDown={handleKeyDown}>
      <td>
        <input
          ref={phraseRef}
          type="text"
          className="input input-sm w-full font-mono"
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          placeholder={t`Phrase`}
        />
      </td>
      <td colSpan={2} className="hidden sm:table-cell">
        <div className="flex items-center gap-3">
          <input
            type="text"
            className="input input-sm flex-1 font-mono"
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            placeholder={t`Replacement`}
          />
          <PriorityPicker value={priority} onChange={setPriority} />
        </div>
      </td>
      <td>
        <div className="sm:hidden mb-2 space-y-2">
          <input
            type="text"
            className="input input-sm w-full font-mono"
            value={replacement}
            onChange={(e) => setReplacement(e.target.value)}
            placeholder={t`Replacement`}
          />
          <PriorityPicker value={priority} onChange={setPriority} />
        </div>
        <div className="flex items-center justify-end gap-1">
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-square"
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
            disabled={saving || !dirty || !phrase.trim() || !replacement.trim()}
            aria-label={t`Save`}
          >
            {saving ? (
              <span className="loading loading-spinner loading-xs" />
            ) : (
              <Check size={15} />
            )}
          </button>
        </div>
      </td>
    </tr>
  );
}

function ViewRow({
  entry,
  onEdit,
}: {
  entry: DictionaryEntry;
  onEdit: () => void;
}) {
  const { t } = useLingui();

  return (
    <tr className="hover group">
      <td>
        <div className="font-mono text-base">{entry.phrase}</div>
        <div className="sm:hidden text-xs text-content-dim mt-0.5">
          → {entry.replacement}
        </div>
      </td>
      <td className="hidden sm:table-cell font-mono text-base">
        {entry.replacement}
      </td>
      <td className="hidden sm:table-cell">
        <PriorityBadge value={entry.priority} />
      </td>
      <td className="text-right">
        <div className="flex items-center justify-end gap-1">
          <span className="sm:hidden">
            <PriorityBadge value={entry.priority} />
          </span>
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-square opacity-60 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity"
            onClick={onEdit}
            aria-label={t`Edit`}
          >
            <Pencil size={14} />
          </button>
          <ConfirmDialog
            title={t`Delete this word fix?`}
            description={t`Remove the saved correction for "${entry.phrase}" → "${entry.replacement}".`}
            onConfirm={() => void removeDictionaryEntry(entry.id)}
          />
        </div>
      </td>
    </tr>
  );
}

export function DictionaryTable() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);
  const [editingId, setEditingId] = useState<number | null>(null);

  if (entries.length === 0) {
    return (
      <div role="alert" className="alert alert-info alert-soft">
        <BookOpen
          size={18}
          strokeWidth={1.5}
          className="text-content-subtle"
        />
        <span>
          <Trans>No entries yet. Add your first correction rule above.</Trans>
        </span>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-box border border-surface-border bg-surface-card">
      <table className="table">
        <thead>
          <tr>
            <th>
              <Trans>Phrase</Trans>
            </th>
            <th className="hidden sm:table-cell">
              <Trans>Replacement</Trans>
            </th>
            <th className="hidden sm:table-cell w-28">
              <Trans>Priority</Trans>
            </th>
            <th className="w-20 sm:w-28 text-right">
              <Trans>Action</Trans>
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) =>
            editingId === entry.id ? (
              <EditRow
                key={entry.id}
                entry={entry}
                onDone={() => setEditingId(null)}
              />
            ) : (
              <ViewRow
                key={entry.id}
                entry={entry}
                onEdit={() => setEditingId(entry.id)}
              />
            ),
          )}
        </tbody>
      </table>
    </div>
  );
}
