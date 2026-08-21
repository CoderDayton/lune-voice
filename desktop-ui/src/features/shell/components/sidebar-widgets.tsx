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
    <div className="border-t border-surface-border bg-base-200">
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
        className="flex w-full cursor-pointer items-center gap-3.5 px-4 py-5 text-left transition-colors hover:bg-surface-hover focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-60 is-drawer-close:justify-center is-drawer-close:px-0"
      >
        <span
          className={`flex size-11 shrink-0 items-center justify-center rounded-field is-drawer-close:size-11 ${
            isActive ? 'bg-error/15' : 'bg-primary/15'
          }`}
        >
          {isActive ? (
            <Square size={18} strokeWidth={2.5} className="text-error" />
          ) : (
            <Mic size={20} strokeWidth={2} className="text-primary" />
          )}
        </span>
        <span className="min-w-0 flex-1 is-drawer-close:hidden">
          <span className={`block text-sm font-medium ${isActive ? 'text-error' : 'text-content'}`}>
            {isActive ? t`Stop dictating` : isConnecting ? t`Connecting` : t`Start dictating`}
          </span>
          {isActive ? (
            // Live level metering lives in the Home "Audio" card — this stays
            // a status line so there's exactly one voice meter on screen.
            <span className="mt-0.5 flex items-center gap-1.5 text-xs text-content-subtle">
              <span className="status status-error animate-pulse" />
              {serviceStatus === 'processing' ? t`Processing…` : t`Listening…`}
            </span>
          ) : (
            <span className="mt-0.5 block text-xs text-content-subtle">{t`Press to talk`}</span>
          )}
        </span>
        {isActive ? null : (
          <kbd className="kbd kbd-sm ml-auto shrink-0 font-mono text-xs is-drawer-close:hidden">
            {hotkeyToggle || t`Set hotkey`}
          </kbd>
        )}
      </button>
    </div>
  );
}
