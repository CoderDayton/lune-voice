import type { ComponentType, SVGProps } from "react";
import {
  DictionaryIcon,
  HistoryIcon,
  OverviewIcon,
  SettingsIcon,
} from "./icons";

export interface NavigationItem {
  to: "/" | "/history" | "/dictionary" | "/settings";
  section: "Use Lune" | "Set up";
  label: string;
  caption: string;
  eyebrow: string;
  description: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

export const navigationItems: NavigationItem[] = [
  {
    to: "/",
    section: "Use Lune",
    label: "Home",
    caption: "Your workspace",
    eyebrow: "Home",
    description: "See your recording status, recent activity, and the fastest next actions.",
    icon: OverviewIcon,
  },
  {
    to: "/history",
    section: "Use Lune",
    label: "Sessions",
    caption: "Recent recordings",
    eyebrow: "Sessions",
    description: "Browse past dictation sessions, search what you said, and revisit saved text.",
    icon: HistoryIcon,
  },
  {
    to: "/dictionary",
    section: "Use Lune",
    label: "Vocabulary",
    caption: "Word fixes",
    eyebrow: "Vocabulary",
    description: "Teach Lune Voice your names, phrases, and preferred spelling so it gets them right.",
    icon: DictionaryIcon,
  },
  {
    to: "/settings",
    section: "Set up",
    label: "Settings",
    caption: "Voice, shortcuts, output",
    eyebrow: "Settings",
    description: "Configure voice input, recognition, cleanup, keyboard shortcuts, and text delivery.",
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
