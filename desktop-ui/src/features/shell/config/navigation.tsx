import type { LucideIcon } from "lucide-react";
import { House, Clock, BookText, Settings } from "lucide-react";

export interface NavigationItem {
  to: "/" | "/history" | "/dictionary" | "/settings";
  label: string;
  icon: LucideIcon;
}

export const navigationItems: NavigationItem[] = [
  {
    to: "/",
    label: "Home",
    icon: House,
  },
  {
    to: "/history",
    label: "Sessions",
    icon: Clock,
  },
  {
    to: "/dictionary",
    label: "Vocabulary",
    icon: BookText,
  },
  {
    to: "/settings",
    label: "Settings",
    icon: Settings,
  },
];

export function findNavigationItem(pathname: string) {
  return (
    navigationItems.find(({ to }) => {
      if (to === "/") {
        return pathname === "/";
      }

      return pathname.startsWith(to);
    }) ?? navigationItems[0]
  );
}
