// SPDX-License-Identifier: GPL-3.0-or-later

import type { RegisterableHotkey } from '@tanstack/react-hotkeys';
import type { LucideIcon } from 'lucide-react';
import { BookText, Clock, House, Paintbrush, ScrollText, Settings } from 'lucide-react';

export type NavigationGroupLabel = 'Workspace' | 'System';

export type NavigationRoute =
  | '/'
  | '/history'
  | '/dictionary'
  | '/snippets'
  | '/styles'
  | '/settings';

/** Single source of truth for the nav hotkeys: `useAppShortcuts` registers
 *  them and the sidebar rail renders them, so hint and binding can't drift.
 *  `Mod` resolves to ⌘ on macOS and Ctrl everywhere else. */
export const navigationShortcuts = {
  '/': 'Mod+1',
  '/history': 'Mod+2',
  '/dictionary': 'Mod+3',
  '/snippets': 'Mod+4',
  '/styles': 'Mod+5',
  '/settings': 'Mod+,',
} as const satisfies Record<NavigationRoute, RegisterableHotkey>;

/** Hotkeys for the sidebar's Quick actions menu. Same single-source rule as the
 *  nav ones: the menu renders these hints and `useAppShortcuts` binds them, so
 *  a hint can never advertise a key that does nothing. */
export const quickActionShortcuts = {
  shortcuts: 'Mod+K',
  quit: 'Mod+Q',
} as const satisfies Record<'shortcuts' | 'quit', RegisterableHotkey>;

export interface NavigationItem {
  to: NavigationRoute;
  label: string;
  /** Short subtitle shown under the page title in the desktop command bar. */
  description: string;
  /** Sidebar grouping — renders as an unlabelled hairline between blocks. */
  group: NavigationGroupLabel;
  shortcut: RegisterableHotkey;
  icon: LucideIcon;
  /** Page owns the viewport height instead of growing with its content: the
   *  frame fills the region and the page's own list is the only scroller.
   *  Falls back to ordinary page scrolling below the `lg` breakpoint. */
  fill?: boolean;
}

export const navigationItems: NavigationItem[] = [
  {
    to: '/',
    label: 'Home',
    description: 'Your dictation at a glance',
    group: 'Workspace',
    shortcut: navigationShortcuts['/'],
    icon: House,
    fill: true,
  },
  {
    to: '/history',
    label: 'Sessions',
    description: 'Browse and search past dictations',
    group: 'Workspace',
    shortcut: navigationShortcuts['/history'],
    icon: Clock,
    fill: true,
  },
  {
    to: '/dictionary',
    label: 'Vocabulary',
    description: 'Corrections Lune Voice applies every time you dictate',
    group: 'Workspace',
    shortcut: navigationShortcuts['/dictionary'],
    icon: BookText,
    fill: true,
  },
  {
    to: '/snippets',
    label: 'Snippets',
    description: 'Blocks of text Lune Voice types when you speak their trigger phrase',
    group: 'Workspace',
    shortcut: navigationShortcuts['/snippets'],
    icon: ScrollText,
    fill: true,
  },
  {
    to: '/styles',
    label: 'Styles',
    description: 'Formatting presets for your text',
    group: 'Workspace',
    shortcut: navigationShortcuts['/styles'],
    icon: Paintbrush,
  },
  {
    to: '/settings',
    label: 'Settings',
    description: 'Input, output, models, and shortcuts',
    group: 'System',
    shortcut: navigationShortcuts['/settings'],
    icon: Settings,
  },
];

export interface NavigationGroup {
  label: NavigationGroupLabel;
  items: NavigationItem[];
}

/** Ordered nav groups for the sidebar. Order here is the render order. */
export const navigationGroups: NavigationGroup[] = (['Workspace', 'System'] as const).map(
  (label) => ({
    label,
    items: navigationItems.filter((item) => item.group === label),
  }),
);

export function findNavigationItem(pathname: string) {
  return (
    navigationItems.find(({ to }) => {
      if (to === '/') {
        return pathname === '/';
      }

      return pathname.startsWith(to);
    }) ?? navigationItems[0]
  );
}
