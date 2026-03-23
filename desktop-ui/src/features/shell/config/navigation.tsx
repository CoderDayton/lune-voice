import type { ComponentType, SVGProps } from "react";
import {
  DictionaryIcon,
  HistoryIcon,
  OverviewIcon,
  SettingsIcon,
} from "./icons";

export interface NavigationItem {
  to: "/" | "/history" | "/dictionary" | "/settings";
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

export const navigationItems: NavigationItem[] = [
  {
    to: "/",
    label: "Home",
    icon: OverviewIcon,
  },
  {
    to: "/history",
    label: "Sessions",
    icon: HistoryIcon,
  },
  {
    to: "/dictionary",
    label: "Vocabulary",
    icon: DictionaryIcon,
  },
  {
    to: "/settings",
    label: "Settings",
    icon: SettingsIcon,
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
