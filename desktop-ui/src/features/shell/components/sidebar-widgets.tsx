import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Mic, Square } from 'lucide-react';
import { toggleDictation, workspaceStore } from '../../../store/workspace-store';

export function SidebarQuickStart() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnecting = serviceStatus === 'Connecting';

  return (
    <div className="px-3 pb-3 sm:px-3.5 is-drawer-close:px-0">
      <button
        type="button"
        onClick={(e) => {
          // Drop focus: dictated text typed into this window must not
          // re-activate the button via Space/Enter and re-toggle sessions.
          e.currentTarget.blur();
          void toggleDictation();
        }}
        disabled={isConnecting}
        title={isActive ? t`Stop dictating` : t`Start dictating`}
        aria-label={isActive ? t`Stop dictating` : t`Start dictating`}
        className={`group flex w-full cursor-pointer items-center gap-4 rounded-field border px-4 py-3.5 shadow-card transition-all sm:gap-4 sm:px-4 sm:py-3.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary is-drawer-close:mx-auto is-drawer-close:size-10 is-drawer-close:justify-center is-drawer-close:gap-0 is-drawer-close:px-0 is-drawer-close:py-0 is-drawer-close:shadow-none ${
          isActive
            ? 'border-error/30 bg-error/[0.08] hover:bg-error/[0.12] hover:border-error/40'
            : 'border-primary/15 bg-primary/[0.06] hover:bg-primary/[0.1] hover:border-primary/25 disabled:cursor-not-allowed disabled:opacity-60'
        }`}
      >
        <div
          className={`flex size-9 shrink-0 items-center justify-center rounded-field is-drawer-close:size-full ${
            isActive ? 'bg-error/15' : 'bg-primary/15'
          }`}
        >
          {isActive ? (
            <Square size={16} strokeWidth={2.5} className="text-error" />
          ) : (
            <Mic size={18} strokeWidth={2} className="text-primary" />
          )}
        </div>
        <div className="min-w-0 text-left is-drawer-close:hidden">
          <span className={`block text-sm font-medium ${isActive ? 'text-error' : 'text-content'}`}>
            {isActive ? t`Stop dictating` : isConnecting ? t`Connecting` : t`Start dictating`}
          </span>
          {isActive ? (
            // Live level metering lives in the Home "Audio" card — this stays
            // a status line so there's exactly one voice meter on screen.
            <span className="mt-1.5 flex items-center gap-1.5 text-xs text-content-subtle">
              <span className="status status-error animate-pulse" />
              {serviceStatus === 'processing' ? t`Processing…` : t`Listening…`}
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
