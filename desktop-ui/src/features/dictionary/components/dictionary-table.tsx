import { useStore } from "@tanstack/react-store";
import { ConfirmDialog } from "../../../components/feedback";
import { removeDictionaryEntry, workspaceStore } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";
import { BookOpen } from "lucide-react";

export function DictionaryTable() {
  const { t } = useLingui();
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  if (entries.length === 0) {
    return (
      <div role="alert" className="alert alert-info alert-soft">
        <BookOpen size={18} strokeWidth={1.5} className="text-content-subtle" />
        <span><Trans>No entries yet. Add your first correction rule above.</Trans></span>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-box border border-surface-border bg-surface-card">
      <table className="table">
        <thead>
          <tr>
            <th><Trans>Phrase</Trans></th>
            <th><Trans>Replacement</Trans></th>
            <th className="w-24 text-right"><Trans>Action</Trans></th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.id} className="hover">
              <td className="font-mono text-base">{entry.phrase}</td>
              <td className="font-mono text-base">{entry.replacement}</td>
              <td className="text-right">
                <ConfirmDialog
                  title={t`Delete this word fix?`}
                  description={`Remove the saved correction for "${entry.phrase}" → "${entry.replacement}".`}
                  onConfirm={() => void removeDictionaryEntry(entry.id)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
