import { Pill } from "../../../components/ui/pill";
import type { NavigationItem } from "../config/navigation";

interface TopbarProps {
  page: NavigationItem;
}

export function Topbar({ page }: TopbarProps) {
  return (
    <header className="topbar-shell">
      <div className="topbar-copy">
        <p className="app-kicker">{page.eyebrow}</p>
        <h1 className="page-title mt-2">{page.label}</h1>
        <p className="body-copy-soft mt-2">{page.description}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Pill>Private on-device</Pill>
        <Pill>Ready to record</Pill>
        <Pill>Works across apps</Pill>
      </div>
    </header>
  );
}
