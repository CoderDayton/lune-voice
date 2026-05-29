import type { LucideIcon } from 'lucide-react';
import { BookText, Clock, House, Paintbrush, Settings } from 'lucide-react';

export interface NavigationItem {
  to: '/' | '/history' | '/dictionary' | '/styles' | '/settings';
  label: string;
  /** Short subtitle shown under the page title in the desktop command bar. */
  description: string;
  icon: LucideIcon;
}

export const navigationItems: NavigationItem[] = [
  {
    to: '/',
    label: 'Home',
    description: 'Your dictation at a glance',
    icon: House,
  },
  {
    to: '/history',
    label: 'Sessions',
    description: 'Browse and search past dictations',
    icon: Clock,
  },
  {
    to: '/dictionary',
    label: 'Vocabulary',
    description: 'Custom phrases and corrections',
    icon: BookText,
  },
  {
    to: '/styles',
    label: 'Styles',
    description: 'Formatting presets for your text',
    icon: Paintbrush,
  },
  {
    to: '/settings',
    label: 'Settings',
    description: 'Input, output, models, and shortcuts',
    icon: Settings,
  },
];

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
