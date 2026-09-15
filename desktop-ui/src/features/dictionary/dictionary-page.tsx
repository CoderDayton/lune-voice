// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { QuickAddRule } from './components/dictionary-form';
import { VocabularyList } from './components/dictionary-table';

/**
 * Vocabulary page. A focused two-section stack: the "add a correction" composer
 * (the primary task) above the searchable list of saved rules. Each child owns
 * its own surface and states.
 *
 * The rules list takes the page's leftover height rather than ending under the
 * last row, so the footer sits at the bottom of the pane and a long list
 * scrolls inside the card instead of scrolling the page.
 */
export function DictionaryPage() {
  const { t } = useLingui();
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack min-h-0 flex-1">
        <div className="panel-surface shrink-0 space-y-4 p-4 sm:p-5">
          <div className="flex items-center gap-3">
            <div className="skeleton-shimmer size-10 rounded-field" />
            <div className="space-y-2">
              <div className="skeleton-shimmer h-4 w-40" />
              <div className="skeleton-shimmer h-3 w-56" />
            </div>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="skeleton-shimmer h-12 flex-1 rounded-field" />
            <div className="skeleton-shimmer h-12 flex-1 rounded-field" />
          </div>
          <div className="skeleton-shimmer h-10 w-full rounded-field" />
        </div>
        <div className="min-h-0 flex-1 divide-y divide-surface-border overflow-hidden">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-4">
              <div className="skeleton-shimmer h-4 w-24 sm:w-32" />
              <div className="skeleton-shimmer h-4 w-24 sm:w-32" />
              <div className="skeleton-shimmer ml-auto h-6 w-16 rounded-field" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack min-h-0 flex-1">
      <QuickAddRule />
      <section aria-label={t`Saved rules`} className="flex min-h-0 flex-1 flex-col">
        <VocabularyList />
      </section>
    </div>
  );
}
