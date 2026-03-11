import { Link, Outlet, createRootRouteWithContext, useRouterState } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface RouterContext { queryClient: QueryClient; }

export const Route = createRootRouteWithContext<RouterContext>()({ component: AppShell });

const NAV = [
  { to: "/",           label: "Overview",   icon: OverviewIcon },
  { to: "/history",    label: "History",    icon: HistoryIcon },
  { to: "/dictionary", label: "Dictionary", icon: DictIcon },
  { to: "/settings",   label: "Settings",   icon: SettingsIcon },
];

function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const router = useRouterState();
  const current = router.location.pathname;

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Ambient glow */}
      <div className="ambient-glow" />

      {/* Sidebar */}
      <aside
        style={{
          width: collapsed ? "var(--sidebar-w-col)" : "var(--sidebar-w)",
          transition: "width 0.2s ease",
          flexShrink: 0,
          background: "var(--bg-1)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          padding: "16px 10px",
          gap: "4px",
          zIndex: 10,
          position: "relative",
        }}
      >
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "4px 6px 16px" }}>
          <LuneLogo />
          {!collapsed && (
            <span style={{ fontWeight: 700, fontSize: "15px", letterSpacing: "-0.01em", color: "var(--fg)" }}>
              Lune Voice
            </span>
          )}
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, display: "flex", flexDirection: "column", gap: "2px" }}>
          {NAV.map(({ to, label, icon: Icon }) => {
            const isActive = to === "/" ? current === "/" : current.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                className={`nav-item${isActive ? " active" : ""}`}
                title={collapsed ? label : undefined}
              >
                <Icon className="nav-icon" />
                {!collapsed && <span>{label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Bottom */}
        <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
          <button
            onClick={() => setCollapsed(c => !c)}
            className="nav-item"
            style={{ border: "none", background: "none", cursor: "pointer", width: "100%" }}
            title={collapsed ? "Expand" : "Collapse"}
          >
            <ChevronIcon collapsed={collapsed} />
            {!collapsed && <span>{collapsed ? "Expand" : "Collapse"}</span>}
          </button>
        </div>
      </aside>

      {/* Main */}
      <main
        className="dot-grid"
        style={{
          flex: 1,
          overflowY: "auto",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ maxWidth: "960px", margin: "0 auto", padding: "40px 32px" }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}

/* ── Icons ──────────────────────────────────────────────────────────────── */

function LuneLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <rect width="28" height="28" rx="8" fill="oklch(0.62 0.19 275)" />
      <path d="M8 20V8l10 12V8" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function OverviewIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="6" height="6" rx="1.5" />
      <rect x="10" y="2" width="6" height="6" rx="1.5" />
      <rect x="2" y="10" width="6" height="6" rx="1.5" />
      <rect x="10" y="10" width="6" height="6" rx="1.5" />
    </svg>
  );
}

function HistoryIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="9" cy="9" r="7" />
      <polyline points="9,5 9,9 12,11" />
    </svg>
  );
}

function DictIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="2" width="12" height="14" rx="2" />
      <line x1="6" y1="6" x2="12" y2="6" />
      <line x1="6" y1="9" x2="12" y2="9" />
      <line x1="6" y1="12" x2="9" y2="12" />
    </svg>
  );
}

function SettingsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="9" cy="9" r="2.5" />
      <path d="M9 2v1.5M9 14.5V16M2 9h1.5M14.5 9H16M4.1 4.1l1.06 1.06M12.84 12.84l1.06 1.06M4.1 13.9l1.06-1.06M12.84 5.16l1.06-1.06" />
    </svg>
  );
}

function ChevronIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg
      width="18" height="18"
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transition: "transform 0.2s ease", transform: collapsed ? "rotate(180deg)" : "none" }}
    >
      <polyline points="11,5 7,9 11,13" />
    </svg>
  );
}
