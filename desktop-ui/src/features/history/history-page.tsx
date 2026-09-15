// SPDX-License-Identifier: GPL-3.0-or-later

import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { HistoryTable } from './components/history-table';

/**
 * Sessions page. A single table card that fills the available viewport so the
 * page never scrolls — the table body scrolls internally, with search and
 * pagination pinned. The height comes from the shell's `fill` layout (see
 * `navigation.ts`), the same way Vocabulary gets it.
 */
export function HistoryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-4 sm:p-5">
        <div className="skeleton-shimmer h-9 w-full shrink-0 rounded-field" />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="skeleton-shimmer h-10 w-full shrink-0 rounded-field" />
        ))}
      </div>
    );
  }

  return <HistoryTable />;
}
