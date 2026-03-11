import { useStore } from "@tanstack/react-store";
import { ModalCancel } from "../../../components/ui/modal-cancel";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { workspaceActions, workspaceStore } from "../../../store/workspace-store";

export function DictionaryTable() {
  const entries = useStore(workspaceStore, (state) => state.dictionaryEntries);

  return (
    <Panel className="overflow-hidden p-0">
      <div className="table-header grid-cols-[1fr_1fr_auto]">
        <span className="table-head-text">Spoken phrase</span>
        <span className="table-head-text">Replacement</span>
        <span className="table-head-text text-right">Action</span>
      </div>

      {entries.length === 0 ? (
        <div className="table-body-empty">No entries yet. Add your first correction rule above.</div>
      ) : (
        <div className="table-list">
          {entries.map((entry) => (
            <div key={entry.id} className="table-row grid-cols-[1fr_1fr_auto]">
              <div className="min-w-0">
                <div className="font-mono supporting-label text-copy-200">{entry.phrase}</div>
                <div className="mt-2">
                  <Pill>Priority {entry.priority}</Pill>
                </div>
              </div>
              <div className="font-mono supporting-label text-copy-50">{entry.replacement}</div>
              <div className="text-right">
                <ModalCancel
                  title="Delete this word fix?"
                  description={`Remove the saved correction for "${entry.phrase}" and "${entry.replacement}".`}
                  onConfirm={() => workspaceActions.removeDictionaryEntry(entry.id)}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
