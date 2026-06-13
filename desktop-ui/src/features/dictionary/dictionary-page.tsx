import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { QuickAddRule } from './components/dictionary-form';
import { VocabularyList } from './components/dictionary-table';

/**
 * Vocabulary page. A focused two-section stack: the "add a correction" composer
 * (the primary task) above the searchable list of saved rules. Each child owns
 * its own surface and states.
 */
export function DictionaryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack">
        <div className="panel-surface space-y-4 p-5 sm:p-6">
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
        <div className="panel-surface divide-y divide-surface-border overflow-hidden">
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
    <div className="page-stack">
      <QuickAddRule />
      <section aria-label="Saved rules">
        <VocabularyList />
      </section>
    </div>
  );
}
