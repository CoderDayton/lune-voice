import { useStore } from "@tanstack/react-store";
import { ModalCancel } from "../../../components/ui/modal-cancel";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { removeDictionaryEntry, workspaceStore } from "../../../store/workspace-store";

export function DictionaryTable() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <Panel className="overflow-hidden p-0">
      <div className="bg-base-300/30 border-b border-base-300/40 grid grid-cols-[1fr_1fr_auto] gap-4 px-6 py-3">
        <span className="label-text-custom">Spoken phrase</span>
        <span className="label-text-custom">Replacement</span>
        <span className="label-text-custom text-right">Action</span>
      </div>

      {entries.length === 0 ? (
        <div className="text-center opacity-50 py-14 px-6 text-sm">No entries yet. Add your first correction rule above.</div>
      ) : (
        <div>
          {entries.map((entry) => (
            <div key={entry.id} className="grid grid-cols-[1fr_1fr_auto] items-center gap-4 px-6 py-3 border-b border-base-300/20 transition hover:bg-base-300/20">
              <div className="min-w-0">
                <div className="font-mono text-sm opacity-80">{entry.phrase}</div>
                <div className="mt-2">
                  <Pill>Priority {entry.priority}</Pill>
                </div>
              </div>
              <div className="font-mono text-sm">{entry.replacement}</div>
              <div className="text-right">
                <ModalCancel
                  title="Delete this word fix?"
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
