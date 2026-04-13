
import { useStore } from "@tanstack/react-store";
import { Mic, Square } from "lucide-react";
import { getTodayStats, workspaceStore } from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function SidebarQuickStart() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const isActive = serviceStatus === "listening" || serviceStatus === "processing";

  return (
    <div className="px-3 pb-3 sm:px-3.5">
      <button
        type="button"
        onClick={() => { import("@tauri-apps/api/core").then(m => m.invoke("toggle_dictation")).catch(() => {}); }}
        className={`group flex w-full items-center gap-3 rounded-field border px-3.5 py-3 transition-all sm:gap-3.5 sm:px-4 sm:py-3.5 ${
          isActive
            ? "border-error/30 bg-error/[0.08] hover:bg-error/[0.12] hover:border-error/40"
            : "border-primary/15 bg-primary/[0.06] hover:bg-primary/[0.1] hover:border-primary/25"
        }`}
      >
        <div className={`flex size-9 items-center justify-center rounded-lg ${
          isActive ? "bg-error/15" : "bg-primary/15"
        }`}>
          {isActive ? (
            <Square size={16} strokeWidth={2.5} className="text-error" />
          ) : (
            <Mic size={18} strokeWidth={2} className="text-primary" />
          )}
        </div>
        <div className="min-w-0 text-left">
          <span className={`block text-sm font-medium ${isActive ? "text-error" : "text-content"}`}>
            {isActive ? t`Stop dictating` : t`Start dictating`}
          </span>
          {isActive ? (
            <span className="flex items-center gap-1.5 text-xs text-content-subtle">
              <span className="status status-error animate-pulse" />
              {serviceStatus === "processing" ? t`Processing…` : t`Listening…`}
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-content-dim">
              <kbd className="kbd kbd-xs">Ctrl</kbd>
              <span>+</span>
              <kbd className="kbd kbd-xs">Space</kbd>
            </span>
          )}
        </div>
      </button>
    </div>
  );
}

export function SidebarTodayStats() {
  const { t } = useLingui();
  const todayStats = useStore(workspaceStore, getTodayStats);

  return (
    <div className="mx-3.5 mb-3 rounded-box border border-surface-border bg-surface-hover px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-widest text-content-dim">{t`Today`}</p>
      <p className="mt-1 text-sm text-content-muted tabular-nums">
        {todayStats.count} {todayStats.count === 1 ? t`session` : t`sessions`}
        <span className="mx-1.5 text-content-dim">·</span>
        {todayStats.words.toLocaleString()} {t`words`}
      </p>
    </div>
  );
}
