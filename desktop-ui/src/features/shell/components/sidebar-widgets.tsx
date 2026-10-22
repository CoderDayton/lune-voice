// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Link } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { Mic, Square } from 'lucide-react';
import { useRef } from 'react';
import { toggleDictation, workspaceStore } from '../../../store/workspace-store';
import { SIDEBAR_NAV_ID } from '../config/drawer';
import { useServiceStatus } from '../hooks/use-service-status';

export function SidebarQuickStart() {
  const { t } = useLingui();
  const { status, isActive, isConnecting } = useServiceStatus();
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const hotkey = hotkeyToggle.trim();
  // Set by Enter/Space on the button and read by the click that follows, so
  // only a keyboard activation moves focus. A mouse press, a script's .click()
  // and assistive-tech activation leave it false.
  const keyboardActivation = useRef(false);

  // The one accent in the sidebar: a panel like the content's own, tinted red
  // while a session is live.
  return (
    <div
      className={`flex items-center rounded-full border transition-colors ${
        isActive ? 'border-error/30 bg-error/10' : 'border-surface-border bg-surface-panel'
      }`}
    >
      <button
        type="button"
        onPointerDown={() => {
          keyboardActivation.current = false;
        }}
        onKeyDown={(e) => {
          keyboardActivation.current = e.key === 'Enter' || e.key === ' ';
        }}
        onClick={(e) => {
          // Dictated text typed into this window must not re-activate the
          // button via Space/Enter and re-toggle sessions, so focus leaves it.
          // A keyboard user is parked on the sidebar's nav, which activates
          // nothing, instead of being dropped to the top of the page.
          const fromKeyboard = keyboardActivation.current;
          keyboardActivation.current = false;
          e.currentTarget.blur();
          if (fromKeyboard) document.getElementById(SIDEBAR_NAV_ID)?.focus();
          void toggleDictation();
        }}
        disabled={isConnecting}
        title={isActive ? t`Stop dictating` : t`Start dictating`}
        aria-label={isActive ? t`Stop dictating` : t`Start dictating`}
        className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-full py-2 pe-3 ps-2 text-start transition-colors hover:bg-surface-hover focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-60 is-drawer-close:flex-none is-drawer-close:justify-center is-drawer-close:px-2"
      >
        <span
          className={`relative flex size-10 shrink-0 items-center justify-center rounded-full ${
            isActive ? 'bg-error' : 'bg-(image:--surface-brand)'
          }`}
        >
          {isActive ? (
            <span
              aria-hidden="true"
              className="animate-pulse-ring absolute -inset-1 rounded-full border border-error/60"
            />
          ) : null}
          {isActive ? (
            <Square size={15} strokeWidth={2.5} className="text-error-content" />
          ) : (
            <Mic size={18} strokeWidth={2} className="text-primary-content" />
          )}
        </span>
        <span className="min-w-0 flex-1 is-drawer-close:hidden">
          <span
            className={`block text-sm font-medium ${isActive ? 'text-content-strong' : 'text-content'}`}
          >
            {isActive ? t`Stop dictating` : isConnecting ? t`Connecting` : t`Start dictating`}
          </span>
          {isActive ? (
            // Live level metering lives in the Home "Audio" card — this stays
            // a status line so there's exactly one voice meter on screen.
            <span className="mt-0.5 flex items-center gap-1.5 text-xs text-content-subtle">
              <span className="status status-error animate-pulse" />
              {status === 'processing' ? t`Processing…` : t`Listening…`}
            </span>
          ) : null}
        </span>
        {isActive || !hotkey ? null : (
          <kbd className="kbd kbd-sm ms-auto shrink-0 font-mono text-xs is-drawer-close:hidden">
            {hotkey}
          </kbd>
        )}
      </button>
      {isActive || hotkey ? null : (
        <Link
          to="/settings"
          search={{ tab: 'dictation' }}
          className="me-2 shrink-0 rounded-full px-2.5 py-1 text-xs text-primary transition-colors hover:bg-surface-hover is-drawer-close:hidden"
        >
          {t`Set hotkey`}
        </Link>
      )}
    </div>
  );
}
