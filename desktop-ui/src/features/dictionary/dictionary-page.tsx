import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { DictionaryForm } from './components/dictionary-form';
import { DictionaryTable } from './components/dictionary-table';

export function DictionaryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="bento-board">
        <div className="panel-surface col-span-12 space-y-3 p-4 sm:p-5 md:col-span-4 md:self-start">
          <div className="skeleton-shimmer h-5 w-40" />
          <div className="skeleton-shimmer h-10 w-full rounded-field" />
          <div className="skeleton-shimmer h-10 w-full rounded-field" />
          <div className="skeleton-shimmer h-10 w-24 rounded-field" />
        </div>
        <div className="panel-surface col-span-12 divide-y divide-surface-border overflow-hidden md:col-span-8">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-4">
              <div className="skeleton-shimmer h-4 w-20 sm:w-28" />
              <div className="skeleton-shimmer h-4 w-20 sm:w-28" />
              <div className="skeleton-shimmer h-6 w-16 ml-auto rounded-field" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bento-board">
      {/* Form rail — sticky on desktop so the table can scroll past it */}
      <div className="col-span-12 md:col-span-4 md:sticky md:top-4 md:self-start">
        <DictionaryForm />
      </div>
      <div className="col-span-12 min-w-0 md:col-span-8">
        <DictionaryTable />
      </div>
    </div>
  );
}
