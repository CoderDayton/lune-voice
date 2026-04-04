import { useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { useLingui } from "@lingui/react/macro";
import { Clock, Hash, MessageSquareText, Mic } from "lucide-react";
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
      <section className="card border border-base-content/[0.06] bg-base-200/40">
        <div className="card-body gap-5">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className={`badge badge-lg gap-2 border-none px-3 ${isLoading ? "badge-warning text-warning-content" : isConnected ? "badge-success text-success-content" : "badge-error text-error-content"}`}>
              <span className={`status ${isLoading ? "status-warning" : isConnected ? "status-success" : "status-error"}`} />
              {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
            </span>
            <span className="text-base-content/45">{summarizeAsrBackend(asrBackend)}</span>
            <span className="hidden text-base-content/25 sm:inline">•</span>
            <span className="text-base-content/45">{summarizeDictationMode(dictationMode)}</span>
          </div>

          {stats.count > 0 ? (
            <div className="stats stats-vertical w-full overflow-hidden rounded-xl border border-base-content/[0.06] bg-base-100/30 sm:stats-horizontal">
              <div className="stat">
                <div className="stat-figure text-primary/30"><Hash size={22} strokeWidth={1.5} /></div>
                <div className="stat-title">{t`Words`}</div>
                <div className="stat-value text-3xl">{stats.totalWords.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="stat-figure text-secondary/30"><Clock size={22} strokeWidth={1.5} /></div>
                <div className="stat-title">{t`Minutes`}</div>
                <div className="stat-value text-3xl">{stats.totalMinutes}</div>
              </div>
              <div className="stat">
                <div className="stat-figure text-accent/30"><MessageSquareText size={22} strokeWidth={1.5} /></div>
                <div className="stat-title">{t`Sessions`}</div>
                <div className="stat-value text-3xl">{stats.count}</div>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-base-content/[0.08] bg-base-100/20 px-5 py-6 text-sm text-base-content/50">
              {t`Your session summary will appear here once you start dictating.`}
            </div>
          )}
        </div>
      </section>

      {isActive && (
        <div className="card border border-primary/15 bg-primary/5" role="region" aria-live="polite" aria-label={t`Live dictation`}>
          <div className="card-body p-4 sm:p-5">
            <div className="mb-2 flex items-center gap-2">
              <span className="loading loading-dots loading-xs text-primary" aria-hidden="true" />
              <span className="text-xs font-semibold uppercase tracking-wider text-primary/70 select-none">
                {serviceStatus === "processing" ? t`Processing` : t`Listening`}
              </span>
            </div>
            <p className="min-h-[2rem] text-sm leading-relaxed sm:text-base">
              {liveTranscript || <span className="italic opacity-30">{t`Waiting for speech…`}</span>}
            </p>
          </div>
        </div>
      )}

      <section className="card border border-base-content/[0.06] bg-base-200/40">
        <div className="card-body p-0">
          <div className="flex items-center justify-between px-5 pb-3 pt-5">
            <div>
              <h2 className="section-label">{t`Recent`}</h2>
              <p className="mt-1 text-sm text-base-content/50">{t`Recent activity`}</p>
            </div>
          {stats.count > 5 && (
              <Link to="/history" className="btn btn-ghost btn-sm normal-case">{t`View all`}</Link>
          )}
          </div>
          {recentSessions.length === 0 && !isActive ? (
            <div className="flex flex-col items-center justify-center px-5 py-14 text-center sm:py-16">
              <div className="mb-4 flex size-16 items-center justify-center rounded-full bg-primary/8 sm:size-20">
                <Mic className="size-8 text-primary/40 sm:size-10" strokeWidth={1.2} />
              </div>
              <p className="text-sm font-medium opacity-70">{t`No sessions yet`}</p>
              <p className="mt-1 text-xs opacity-40">
                {t`Press`} <kbd className="kbd kbd-xs">{t`Ctrl`}</kbd> + <kbd className="kbd kbd-xs">{t`Space`}</kbd> {t`to start dictating`}
              </p>
            </div>
          ) : recentSessions.length === 0 && isActive ? null : (
            <div className="divide-y divide-base-content/[0.06]">
              {recentSessions.map((session) => (
                <article key={session.id} className="px-5 py-4">
                  <div className="pt-1 text-xs tabular-nums text-base-content/40">{session.time}</div>
                  <div className="min-w-0">
                    <p className="text-sm leading-6 break-words text-base-content/92">
                      {session.transcript || t`Empty session`}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-base-content/50">
                      <span className="badge badge-ghost badge-sm">{session.words} {t`words`}</span>
                      <span>{session.seconds}{t`s`}</span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
