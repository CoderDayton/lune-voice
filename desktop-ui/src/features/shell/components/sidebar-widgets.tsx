import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { Mic } from "lucide-react";
import { workspaceStore } from "../../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function SidebarQuickStart() {
  const { t } = useLingui();

  return (
    <div className="px-3 pb-3 sm:px-3.5">
      <button
        type="button"
        onClick={() => { import("@tauri-apps/api/core").then(m => m.invoke("toggle_dictation")).catch(() => {}); }}
        className="group flex w-full items-center gap-3 rounded-xl border border-primary/15 bg-primary/[0.06] px-3.5 py-3 transition-colors hover:bg-primary/[0.1] hover:border-primary/25 sm:gap-3.5 sm:px-4 sm:py-3.5"
      >
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/15">
          <Mic size={18} strokeWidth={2} className="text-primary" />
        </div>
        <div className="min-w-0 text-left">
          <span className="block text-sm font-medium text-base-content/80">{t`Start dictating`}</span>
          <span className="flex items-center gap-1 text-xs text-base-content/35">
            <kbd className="kbd kbd-xs">Ctrl</kbd>
            <span>+</span>
            <kbd className="kbd kbd-xs">Space</kbd>
          </span>
        </div>
      </button>
    </div>
  );
}

export function SidebarTodayStats() {
  const { t } = useLingui();
  const sessions = useStore(workspaceStore, (s) => s.historySessions);

  const todayStats = useMemo(() => {
    const today = new Date().toDateString();
    const todaySessions = sessions.filter((s) => new Date(s.startedAt).toDateString() === today);
    return {
      count: todaySessions.length,
      words: todaySessions.reduce((sum, s) => sum + s.words, 0),
    };
  }, [sessions]);

  if (todayStats.count === 0) return null;

  return (
    <div className="mx-3.5 mb-3 rounded-xl border border-base-content/[0.05] bg-base-content/[0.02] px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-widest text-base-content/30">{t`Today`}</p>
      <p className="mt-1 text-sm text-base-content/55 tabular-nums">
        {todayStats.count} {todayStats.count === 1 ? t`session` : t`sessions`}
        <span className="mx-1.5 text-base-content/20">·</span>
        {todayStats.words.toLocaleString()} {t`words`}
      </p>
    </div>
  );
}
