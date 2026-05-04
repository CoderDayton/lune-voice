import { useStore } from '@tanstack/react-store';
import { workspaceStore } from '../../store/workspace-store';
import { DictionaryForm } from './components/dictionary-form';
import { DictionaryTable } from './components/dictionary-table';

export function DictionaryPage() {
  const loaded = useStore(workspaceStore, (s) => s.runtime.loaded);

  if (!loaded) {
    return (
      <div className="page-stack">
        <div className="card border rounded-box p-4 sm:p-5 border-surface-border bg-surface-card space-y-3">
          <div className="skeleton h-5 w-40" />
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="skeleton h-10 flex-1 rounded-field" />
            <div className="skeleton h-10 flex-1 rounded-field" />
            <div className="skeleton h-10 w-full sm:w-20 rounded-field" />
          </div>
        </div>
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex items-center gap-3 py-2">
              <div className="skeleton h-4 w-20 sm:w-28" />
              <div className="skeleton h-4 w-20 sm:w-28" />
              <div className="skeleton h-6 w-16 ml-auto rounded-field" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <DictionaryForm />
      <DictionaryTable />
    </div>
  );
}
