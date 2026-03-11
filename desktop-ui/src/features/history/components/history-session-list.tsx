import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { Panel } from "../../../components/ui/panel";
import { Pill } from "../../../components/ui/pill";
import { historyGroups } from "../data/history-data";
import { workspaceStore } from "../../../store/workspace-store";

export function HistorySessionList() {
  const search = useStore(workspaceStore, (state) => state.historySearch);

  const filteredGroups = useMemo(() => {
    return historyGroups
      .map((group) => ({
        ...group,
        items: group.items.filter((item) => !search || item.text.toLowerCase().includes(search.toLowerCase())),
      }))
      .filter((group) => group.items.length > 0);
  }, [search]);

  return (
    <Panel className="overflow-hidden p-0">
      {filteredGroups.map((group) => (
        <section key={group.date}>
          <div className="session-group-head">
            <span className="session-group-label">{group.date}</span>
          </div>

          <div className="table-list">
            {group.items.map((item) => (
              <article key={`${group.date}-${item.time}-${item.text}`} className="session-row">
                <div className="flex flex-wrap gap-2">
                  <Pill>{item.time}</Pill>
                  <Pill>{item.words} words</Pill>
                  <Pill>{item.seconds}s audio</Pill>
                  <Pill>{item.mode}</Pill>
                </div>
                <p className="session-copy mt-4">{item.text}</p>
              </article>
            ))}
          </div>
        </section>
      ))}
    </Panel>
  );
}
