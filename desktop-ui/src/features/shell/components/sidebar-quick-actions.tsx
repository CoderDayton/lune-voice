import { useLingui } from '@lingui/react/macro';
import { formatForDisplay } from '@tanstack/react-hotkeys';
import { useRouter } from '@tanstack/react-router';
import {
  ChevronUp,
  CircleQuestionMark,
  Keyboard,
  Power,
  RefreshCw,
  SlidersHorizontal,
  Sun,
} from 'lucide-react';
import { useRef, useState } from 'react';
import { HelpFeedbackDialog } from '../../../components/feedback/help-feedback-dialog';
import { ActionMenu, type ActionMenuItem } from '../../../components/ui/action-menu';
import { ThemePickerDialog } from '../../../components/ui/theme-picker';
import { backendClient } from '../../../lib/backend/client';
import { getThemePreference, themeLabel } from '../../../lib/theme';
import { workspaceActions } from '../../../store/workspace-store';
import { DRAWER_ID } from '../config/drawer';
import { quickActionShortcuts } from '../config/navigation';

/** Surface a fire-and-forget command failure instead of leaving it as an
 *  invisible unhandled rejection in the webview console. */
function reportFailure(error: unknown) {
  workspaceActions.setError(error instanceof Error ? error.message : String(error));
}

/** The drawer checkbox is checked while the sidebar is expanded. */
function isSidebarExpanded(): boolean {
  const toggle = document.getElementById(DRAWER_ID);
  return toggle instanceof HTMLInputElement ? toggle.checked : true;
}

export function SidebarQuickActions() {
  const { t } = useLingui();
  const router = useRouter();
  const triggerRef = useRef<HTMLButtonElement>(null);

  const [open, setOpen] = useState(false);
  const [placement, setPlacement] = useState<'top' | 'right'>('top');
  const [themeName, setThemeName] = useState('');
  const [themeOpen, setThemeOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  // Both are read at open time rather than tracked continuously: neither the
  // collapsed state nor the theme can change while the menu is up.
  function toggleMenu() {
    if (!open) {
      setPlacement(isSidebarExpanded() ? 'top' : 'right');
      const pref = getThemePreference();
      setThemeName(pref === 'system' ? t`System` : themeLabel(pref));
    }
    setOpen((prev) => !prev);
  }

  const items: ActionMenuItem[] = [
    {
      id: 'theme',
      label: t`Change theme`,
      icon: Sun,
      trailing: themeName,
      onSelect: () => setThemeOpen(true),
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
    <div className="border-t border-surface-border bg-base-200">
      <button
        ref={triggerRef}
        type="button"
        onClick={toggleMenu}
        aria-haspopup="menu"
        aria-expanded={open}
        title={t`Quick actions`}
        aria-label={t`Quick actions`}
        className="flex w-full cursor-pointer items-center gap-3.5 px-4 py-4 text-left transition-colors hover:bg-surface-hover focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary is-drawer-close:justify-center is-drawer-close:px-0"
      >
        <span className="flex size-11 shrink-0 items-center justify-center rounded-field bg-surface-elevated ring-1 ring-surface-border is-drawer-close:size-11">
          <SlidersHorizontal size={18} strokeWidth={1.8} className="text-content-subtle" />
        </span>
        <span className="min-w-0 flex-1 is-drawer-close:hidden">
          <span className="block text-sm font-medium text-content">{t`Quick actions`}</span>
          <span className="mt-0.5 block text-xs text-content-subtle">
            {t`Theme, shortcuts, quit`}
          </span>
        </span>
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
        ariaLabel={t`Quick actions`}
        open={open}
        onOpenChange={setOpen}
        anchorRef={triggerRef}
        placement={placement}
      />

      <ThemePickerDialog open={themeOpen} onClose={() => setThemeOpen(false)} />
      <HelpFeedbackDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
