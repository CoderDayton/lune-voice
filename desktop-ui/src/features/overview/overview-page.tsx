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
  const runtime = useStore(workspaceStore, (s) => s.runtime);
  const settings = useStore(workspaceStore, (s) => s.settings);
  const sessions = useStore(workspaceStore, (s) => s.historySessions);

  const totalWords = sessions.reduce((sum, s) => sum + s.words, 0);
  const totalSeconds = sessions.reduce((sum, s) => sum + s.seconds, 0);
  const totalMinutes = Math.round(totalSeconds / 60);

  const isConnected = runtime.serviceStatus === "ready" || runtime.serviceStatus === "listening";
  const isLoading = runtime.serviceStatus === "Connecting" || runtime.loading;

  return (
    <div className="page-stack">
      {/* Status */}
      <div className="flex items-center gap-3 text-sm">
        <span className={`status ${isLoading ? "status-warning" : isConnected ? "status-success" : "status-error"}`} />
        <span className="opacity-70">
          {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
        </span>
        <span className="opacity-30">·</span>
        <span className="opacity-50">{summarizeAsrBackend(settings.asrBackend)}</span>
        <span className="opacity-30">·</span>
        <span className="opacity-50">{summarizeDictationMode(settings.dictationMode)}</span>
      </div>

      {/* Stats */}
      {sessions.length > 0 && (
        <div className="stats stats-vertical sm:stats-horizontal bg-base-200 w-full">
          <div className="stat">
            <div className="stat-title">{t`Words`}</div>
            <div className="stat-value text-xl">{totalWords.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="stat-title">{t`Minutes`}</div>
            <div className="stat-value text-xl">{totalMinutes}</div>
          </div>
          <div className="stat">
            <div className="stat-title">{t`Sessions`}</div>
            <div className="stat-value text-xl">{sessions.length}</div>
          </div>
        </div>
      )}

      {/* Recent */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider opacity-50">{t`Recent`}</h2>
          {sessions.length > 5 && (
            <Link to="/history" className="link link-primary text-xs">{t`View all`}</Link>
          )}
        </div>
        {sessions.length === 0 ? (
          <div className="alert">
            <span>{t`No sessions yet. Press Ctrl+Space to start dictating.`}</span>
          </div>
        ) : (
          <ul className="list bg-base-200 rounded-box">
            {sessions.slice(0, 5).map((session) => (
              <li key={session.id} className="list-row">
                <div className="text-xs opacity-40 tabular-nums">
                  {new Date(session.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
                </div>
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
