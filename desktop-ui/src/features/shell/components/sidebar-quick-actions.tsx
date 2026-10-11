// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { formatForDisplay } from '@tanstack/react-hotkeys';
import { useRouter } from '@tanstack/react-router';
import {
  ChevronUp,
  CircleQuestionMark,
  Ellipsis,
  Keyboard,
  Moon,
  Power,
  RefreshCw,
  Sun,
} from 'lucide-react';
import { useRef, useState } from 'react';
import { HelpFeedbackDialog } from '../../../components/feedback/help-feedback-dialog';
import { ActionMenu, type ActionMenuItem } from '../../../components/ui/action-menu';
import { NavBead } from '../../../components/ui/nav-item';
import { backendClient } from '../../../lib/backend/client';
import { setThemeMode, toggledThemeMode, useThemeMode } from '../../../lib/theme';
import { workspaceActions } from '../../../store/workspace-store';
import { isSidebarExpanded } from '../config/drawer';
import { quickActionShortcuts } from '../config/navigation';

/** Surface a fire-and-forget command failure instead of leaving it as an
 *  invisible unhandled rejection in the webview console. */
function reportFailure(error: unknown) {
  workspaceActions.setError(error instanceof Error ? error.message : String(error));
}

export function SidebarQuickActions() {
  const { t } = useLingui();
  const router = useRouter();
  const triggerRef = useRef<HTMLButtonElement>(null);

  const [open, setOpen] = useState(false);
  const [placement, setPlacement] = useState<'top' | 'right'>('top');
  const themeMode = useThemeMode();
  const [helpOpen, setHelpOpen] = useState(false);

  // Read at open time rather than tracked: the collapsed state cannot change
  // while the menu is up.
  function toggleMenu() {
    if (!open) setPlacement(isSidebarExpanded() ? 'top' : 'right');
    setOpen((prev) => !prev);
  }

  // One click always changes the look; "System" is chosen in Settings.
  const targetMode = toggledThemeMode(
    themeMode,
    window.matchMedia('(prefers-color-scheme: light)').matches,
  );

  const items: ActionMenuItem[] = [
    {
      id: 'theme',
      label: targetMode === 'light' ? t`Switch to light theme` : t`Switch to dark theme`,
      icon: targetMode === 'light' ? Sun : Moon,
      onSelect: () => setThemeMode(targetMode),
    },
    {
      id: 'shortcuts',
      label: t`Keyboard shortcuts`,
      icon: Keyboard,
      trailing: formatForDisplay(quickActionShortcuts.shortcuts),
      onSelect: () => {
        void router.navigate({ to: '/settings', search: { tab: 'dictation' } });
      },
    },
    {
      id: 'updates',
      label: t`Check for updates`,
      icon: RefreshCw,
      // The outcome — up to date, an offer, or a failure — arrives in the
      // shell's update modal. Only a check that never started is reported here.
      onSelect: () => {
        void backendClient.startUpdateCheck().catch(reportFailure);
      },
    },
    {
      id: 'help',
      label: t`Help & feedback`,
      icon: CircleQuestionMark,
      onSelect: () => setHelpOpen(true),
    },
    {
      id: 'quit',
      label: t`Quit Lune Voice`,
      icon: Power,
      trailing: formatForDisplay(quickActionShortcuts.quit),
      danger: true,
      separatorBefore: true,
      onSelect: () => {
        void backendClient.quitApp().catch(reportFailure);
      },
    },
  ];

  return (
    <>
      {/* A plain row, so the mic panel above stays the sidebar's one accent. */}
      <button
        ref={triggerRef}
        type="button"
        onClick={toggleMenu}
        aria-haspopup="menu"
        aria-expanded={open}
        title={t`More`}
        aria-label={t`More`}
        className="group flex h-12 w-full cursor-pointer items-center gap-3 rounded-full pr-4 pl-2 text-left text-sm text-content-muted transition-colors hover:bg-surface-hover hover:text-content focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary is-drawer-close:justify-center is-drawer-close:px-0"
      >
        <NavBead active={false}>
          <Ellipsis
            size={17}
            strokeWidth={1.8}
            className="text-content-subtle transition-colors group-hover:text-content-muted"
          />
        </NavBead>
        <span className="min-w-0 flex-1 truncate is-drawer-close:hidden">{t`More`}</span>
        {/* Stays pointing up in both states: it marks which way the menu
            opens, rather than flipping to mean "close". */}
        <ChevronUp
          size={16}
          strokeWidth={1.8}
          className="ml-auto shrink-0 text-content-subtle is-drawer-close:hidden"
        />
      </button>

      <ActionMenu
        items={items}
        ariaLabel={t`More`}
        open={open}
        onOpenChange={setOpen}
        anchorRef={triggerRef}
        placement={placement}
      />

      <HelpFeedbackDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </>
  );
}
