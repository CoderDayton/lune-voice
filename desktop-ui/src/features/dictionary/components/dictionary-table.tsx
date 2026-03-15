import { useStore } from "@tanstack/react-store";
import { ModalCancel } from "../../../components/ui/modal-cancel";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { removeDictionaryEntry, workspaceStore } from "../../../store/workspace-store";
import { Trans, useLingui } from "@lingui/react/macro";

export function DictionaryTable() {
  const { t } = useLingui();
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <Panel className="overflow-x-auto p-0">
      <div className="bg-base-300/30 border-b border-base-300/40 grid grid-cols-[1fr_auto] sm:grid-cols-[1fr_1fr_auto] gap-4 px-4 sm:px-6 py-3">
        <span className="label-text-custom"><Trans>Phrase</Trans></span>
        <span className="label-text-custom hidden sm:block"><Trans>Replacement</Trans></span>
        <span className="label-text-custom text-right"><Trans>Action</Trans></span>
      </div>

      {entries.length === 0 ? (
        <div className="text-center opacity-50 py-14 px-6 text-sm"><Trans>No entries yet. Add your first correction rule above.</Trans></div>
      ) : (
        <div>
          {entries.map((entry) => (
            <div key={entry.id} className="grid grid-cols-[1fr_auto] sm:grid-cols-[1fr_1fr_auto] items-center gap-4 px-4 sm:px-6 py-3 border-b border-base-300/20 transition hover:bg-base-300/20">
              <div className="min-w-0">
                <div className="font-mono text-sm opacity-80 truncate">{entry.phrase}</div>
                <div className="font-mono text-sm sm:hidden mt-1">→ {entry.replacement}</div>
                <div className="mt-2">
                  <Pill><Trans>Priority {entry.priority}</Trans></Pill>
                </div>
              </div>
              <div className="font-mono text-sm hidden sm:block">{entry.replacement}</div>
              <div className="text-right">
                <ModalCancel
                  title={t`Delete this word fix?`}
                  description={`Remove the saved correction for "${entry.phrase}" and "${entry.replacement}".`}
                  onConfirm={() => void removeDictionaryEntry(entry.id)}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
