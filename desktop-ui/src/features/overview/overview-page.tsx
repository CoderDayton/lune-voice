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

  const liveTranscript = useStore(workspaceStore, (s) => s.runtime.liveTranscript);
  const isActive = serviceStatus === "listening" || serviceStatus === "processing";
  const isConnected =
    serviceStatus === "ready" ||
    serviceStatus === "listening" ||
    serviceStatus === "processing";
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

      {isActive && (
        <div className="card bg-primary/5 border border-primary/15 rounded-box p-4 sm:p-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="loading loading-dots loading-xs text-primary" />
            <span className="text-xs font-semibold uppercase tracking-wider text-primary/70 select-none">
              {serviceStatus === "processing" ? t`Processing` : t`Listening`}
            </span>
          </div>
          <p className="text-sm sm:text-base leading-relaxed min-h-[2rem]">
            {liveTranscript || <span className="opacity-30 italic">{t`Waiting for speech…`}</span>}
          </p>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider opacity-50">{t`Recent`}</h2>
          {stats.count > 5 && (
            <Link to="/history" className="link link-primary text-xs">{t`View all`}</Link>
          )}
        </div>
        {recentSessions.length === 0 && !isActive ? (
          <div className="flex flex-col items-center justify-center py-12 sm:py-16 text-center">
            <div className="size-16 sm:size-20 rounded-full bg-primary/8 flex items-center justify-center mb-4">
              <svg className="size-8 sm:size-10 text-primary/40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="7" y="3" width="10" height="14" rx="5" />
                <path d="M5 10v1a7 7 0 0 0 14 0v-1M12 17v4M9 21h6" />
              </svg>
            </div>
            <p className="text-sm font-medium opacity-70">{t`No sessions yet`}</p>
            <p className="text-xs opacity-40 mt-1">
              {t`Press`} <kbd className="kbd kbd-xs">{t`Ctrl`}</kbd> + <kbd className="kbd kbd-xs">{t`Space`}</kbd> {t`to start dictating`}
            </p>
          </div>
        ) : recentSessions.length === 0 && isActive ? null : (
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
