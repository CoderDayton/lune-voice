import type { ComponentType, SVGProps } from "react";
import {
  BoltIcon,
  DictionaryIcon,
  HistoryIcon,
  MicIcon,
  OverviewIcon,
} from "./icons";

export interface NavigationItem {
  to: "/" | "/history" | "/dictionary" | "/recording" | "/shortcuts";
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
    to: "/recording",
    section: "Set up",
    label: "Recording",
    caption: "Voice typing setup",
    eyebrow: "Recording",
    description: "Choose how Lune Voice listens, cleans up speech, and turns it into text.",
    icon: MicIcon,
  },
  {
    to: "/shortcuts",
    section: "Set up",
    label: "Shortcuts",
    caption: "Controls and sending",
    eyebrow: "Shortcuts",
    description: "Manage keyboard shortcuts, start and stop behavior, and how text is sent into apps.",
    icon: BoltIcon,
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
