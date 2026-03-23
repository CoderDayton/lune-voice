import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { useLingui } from "@lingui/react/macro";
import {
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from "../../store/workspace-store";

export function OverviewPage() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const dictationMode = useStore(workspaceStore, (s) => s.settings.dictationMode);
  const sessions = useStore(workspaceStore, (s) => s.historySessions);

  const isConnected = serviceStatus === "ready" || serviceStatus === "listening";
  const isLoading = serviceStatus === "Connecting" || loading;

  const stats = useMemo(() => {
    const totalWords = sessions.reduce((sum, s) => sum + s.words, 0);
    const totalSeconds = sessions.reduce((sum, s) => sum + s.seconds, 0);
    return { totalWords, totalMinutes: Math.round(totalSeconds / 60), count: sessions.length };
  }, [sessions]);

  const recentSessions = useMemo(() =>
    sessions.slice(0, 5).map((s) => ({
      id: s.id,
      time: new Date(s.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
      transcript: s.transcript,
      words: s.words,
      seconds: s.seconds,
    })),
  [sessions]);

  return (
    <div className="page-stack">
      <div className="flex items-center gap-3 text-sm">
        <span className={`status ${isLoading ? "status-warning" : isConnected ? "status-success" : "status-error"}`} />
        <span className="opacity-70">
          {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
        </span>
        <span className="opacity-30">·</span>
        <span className="opacity-50">{summarizeAsrBackend(asrBackend)}</span>
        <span className="opacity-30">·</span>
        <span className="opacity-50">{summarizeDictationMode(dictationMode)}</span>
      </div>

      {stats.count > 0 && (
        <div className="stats stats-vertical sm:stats-horizontal bg-base-200 w-full">
          <div className="stat">
            <div className="stat-title">{t`Words`}</div>
            <div className="stat-value text-xl">{stats.totalWords.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="stat-title">{t`Minutes`}</div>
            <div className="stat-value text-xl">{stats.totalMinutes}</div>
          </div>
          <div className="stat">
            <div className="stat-title">{t`Sessions`}</div>
            <div className="stat-value text-xl">{stats.count}</div>
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider opacity-50">{t`Recent`}</h2>
          {stats.count > 5 && (
            <Link to="/history" className="link link-primary text-xs">{t`View all`}</Link>
          )}
        </div>
        {recentSessions.length === 0 ? (
          <div className="alert">
            <span>{t`No sessions yet. Press Ctrl+Space to start dictating.`}</span>
          </div>
        ) : (
          <ul className="list bg-base-200 rounded-box">
            {recentSessions.map((session) => (
              <li key={session.id} className="list-row">
                <div className="text-xs opacity-40 tabular-nums">{session.time}</div>
                <div>
                  <div className="text-sm truncate">{session.transcript || t`Empty session`}</div>
                  <div className="text-xs opacity-40">{session.words} {t`words`} · {session.seconds}{t`s`}</div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
