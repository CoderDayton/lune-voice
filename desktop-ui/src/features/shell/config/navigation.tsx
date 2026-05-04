import type { LucideIcon } from 'lucide-react';
import { BookText, Clock, House, Paintbrush, Settings } from 'lucide-react';

export interface NavigationItem {
  to: '/' | '/history' | '/dictionary' | '/styles' | '/settings';
  label: string;
  icon: LucideIcon;
}

export const navigationItems: NavigationItem[] = [
  {
    to: '/',
    label: 'Home',
    icon: House,
  },
  {
    to: '/history',
    label: 'Sessions',
    icon: Clock,
  },
  {
    to: '/dictionary',
    label: 'Vocabulary',
    icon: BookText,
  },
  {
    to: '/styles',
    label: 'Styles',
    icon: Paintbrush,
  },
  {
    to: '/settings',
    label: 'Settings',
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
