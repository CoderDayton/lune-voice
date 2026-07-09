import type { LucideIcon } from 'lucide-react';
import { BookText, Clock, House, Paintbrush, ScrollText, Settings } from 'lucide-react';

export type NavigationGroupLabel = 'Workspace' | 'System';

export interface NavigationItem {
  to: '/' | '/history' | '/dictionary' | '/snippets' | '/styles' | '/settings';
  label: string;
  /** Short subtitle shown under the page title in the desktop command bar. */
  description: string;
  /** Sidebar grouping — mirrors the shadcn sidebar-07 section pattern. */
  group: NavigationGroupLabel;
  icon: LucideIcon;
}

export const navigationItems: NavigationItem[] = [
  {
    to: '/',
    label: 'Home',
    description: 'Your dictation at a glance',
    group: 'Workspace',
    icon: House,
  },
  {
    to: '/history',
    label: 'Sessions',
    description: 'Browse and search past dictations',
    group: 'Workspace',
    icon: Clock,
  },
  {
    to: '/dictionary',
    label: 'Vocabulary',
    description: 'Custom phrases and corrections',
    group: 'Workspace',
    icon: BookText,
  },
  {
    to: '/snippets',
    label: 'Snippets',
    description: 'Templates you expand by voice',
    group: 'Workspace',
    icon: ScrollText,
  },
  {
    to: '/styles',
    label: 'Styles',
    description: 'Formatting presets for your text',
    group: 'Workspace',
    icon: Paintbrush,
  },
  {
    to: '/settings',
    label: 'Settings',
    description: 'Input, output, models, and shortcuts',
    group: 'System',
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
