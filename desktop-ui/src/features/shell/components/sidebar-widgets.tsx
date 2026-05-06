import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Mic, Square } from 'lucide-react';
import { getTodayStats, toggleDictation, workspaceStore } from '../../../store/workspace-store';

export function SidebarQuickStart() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const audioLevel = useStore(workspaceStore, (s) => s.runtime.audioLevel);
  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnecting = serviceStatus === 'Connecting';
  const level = `${Math.round(audioLevel * 100)}%`;

  return (
    <div className="px-3 pb-3 sm:px-3.5">
      <button
        type="button"
        onClick={() => void toggleDictation()}
        disabled={isConnecting}
        className={`group flex w-full items-center gap-4 rounded-field border px-4 py-3.5 shadow-card transition-all sm:gap-4 sm:px-4 sm:py-3.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${
          isActive
            ? 'border-error/30 bg-error/[0.08] hover:bg-error/[0.12] hover:border-error/40'
            : 'border-primary/15 bg-primary/[0.06] hover:bg-primary/[0.1] hover:border-primary/25 disabled:cursor-not-allowed disabled:opacity-60'
        }`}
      >
        <div
          className={`flex size-9 items-center justify-center rounded-lg ${
            isActive ? 'bg-error/15' : 'bg-primary/15'
          }`}
        >
          {isActive ? (
            <Square size={16} strokeWidth={2.5} className="text-error" />
          ) : (
            <Mic size={18} strokeWidth={2} className="text-primary" />
          )}
        </div>
        <div className="min-w-0 text-left">
          <span className={`block text-sm font-medium ${isActive ? 'text-error' : 'text-content'}`}>
            {isActive ? t`Stop dictating` : isConnecting ? t`Connecting` : t`Start dictating`}
          </span>
          {isActive ? (
            <span className="mt-1.5 block">
              <span className="flex items-center gap-1.5 text-xs text-content-subtle">
                <span className="status status-error animate-pulse" />
                {serviceStatus === 'processing' ? t`Processing…` : t`Listening…`}
              </span>
              <span className="mt-2 block h-1.5 overflow-hidden rounded-full bg-base-content/10">
                <span
                  className="block h-full rounded-full bg-error transition-[width] duration-75"
                  style={{ width: level }}
                />
              </span>
            </span>
          ) : (
            <span className="flex min-w-0 items-center gap-1 text-xs text-content-dim">
              <kbd className="kbd kbd-xs truncate">{hotkeyToggle || t`Set hotkey`}</kbd>
            </span>
          )}
        </div>
      </button>
    </div>
  );
}

export function SidebarTodayStatsSkeleton() {
  return (
    <div className="px-3 py-2 sm:px-4">
      <div className="skeleton-shimmer h-3 w-12 mb-1.5" />
      <div className="skeleton-shimmer h-3 w-28 sm:w-36" />
    </div>
  );
}

export function SidebarTodayStats() {
  const { t } = useLingui();
  const todayStats = useStore(workspaceStore, getTodayStats);

  return (
    <div className="mx-3.5 mb-3 rounded-box border border-surface-border bg-surface-elevated px-4 py-3 shadow-card">
      <p className="text-xs font-medium uppercase tracking-widest text-content-dim">{t`Today`}</p>
      <p className="mt-1 text-sm text-content-muted tabular-nums">
        {todayStats.count} {todayStats.count === 1 ? t`session` : t`sessions`}
        <span className="mx-1.5 text-content-dim">·</span>
        {todayStats.words.toLocaleString()} {t`words`}
      </p>
    </div>
  );
}
