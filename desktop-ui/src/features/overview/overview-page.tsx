import { useStore } from "@tanstack/react-store";
import { Link } from "@tanstack/react-router";
import { SessionDetailModal } from "../history/components/session-detail-modal";
import { ModelSetup } from "./components/model-setup";
import { useLingui } from "@lingui/react/macro";
import { Clock, Hash, MessageSquareText, Mic } from "lucide-react";
import { StatusBadge, InfoBadge } from "../../components/ui/status-badge";
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

  const isActive = serviceStatus === "listening" || serviceStatus === "processing";
  const isConnected =
    serviceStatus === "ready" ||
    serviceStatus === "listening" ||
    serviceStatus === "processing";
  const isLoading = serviceStatus === "Connecting" || loading;
  const showSkeleton = loading && sessions.length === 0;

  const totalWords = sessions.reduce((sum, s) => sum + s.words, 0);
  const totalSeconds = sessions.reduce((sum, s) => sum + s.seconds, 0);
  const stats = { totalWords, totalMinutes: Math.round(totalSeconds / 60), count: sessions.length };

  const recentSessions = sessions.slice(0, 5);

  return (
    <ModelSetup>
    <div className="page-stack">
      <section className="card border border-surface-border bg-surface-card">
        <div className="card-body gap-5">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <StatusBadge variant={isLoading ? "warning" : isConnected ? "success" : "error"}>
              {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
            </StatusBadge>
            <InfoBadge>{summarizeAsrBackend(asrBackend)}</InfoBadge>
            <InfoBadge>{summarizeDictationMode(dictationMode)}</InfoBadge>
          </div>

          {showSkeleton ? (
            <div className="stats stats-vertical w-full overflow-hidden rounded-box border border-surface-border bg-surface-card sm:stats-horizontal">
              {[0, 1, 2].map((i) => (
                <div key={i} className="stat">
                  <div className="stat-title"><span className="skeleton h-3 w-12" /></div>
                  <div className="stat-value"><span className="skeleton h-8 w-20 mt-1" /></div>
                </div>
              ))}
            </div>
          ) : stats.count > 0 ? (
            <div className="stats stats-vertical w-full overflow-hidden rounded-box border border-surface-border bg-surface-card sm:stats-horizontal">
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
            <div className="rounded-box border border-dashed border-surface-border bg-surface-inset px-5 py-6 text-sm text-content-muted">
              {t`Your session summary will appear here once you start dictating.`}
            </div>
          )}
        </div>
      </section>

      <section className="card border border-surface-border bg-surface-card">
        <div className="card-body p-0">
          <div className="flex items-center justify-between px-5 pb-3 pt-5">
            <div>
              <h2 className="section-label">{t`Recent`}</h2>
              <p className="mt-1 text-sm text-content-muted">{stats.count} {stats.count === 1 ? t`session` : t`sessions`}</p>
            </div>
          {stats.count > 5 && (
              <Link to="/history" className="btn btn-ghost btn-sm normal-case">{t`View all`}</Link>
          )}
          </div>
          {showSkeleton ? (
            <div className="divide-y divide-surface-border">
              {[0, 1, 2].map((i) => (
                <div key={i} className="px-5 py-4 flex flex-col gap-2">
                  <span className="skeleton h-3 w-16" />
                  <span className="skeleton h-4 w-full" />
                  <span className="skeleton h-4 w-3/4" />
                  <span className="skeleton h-3 w-24 mt-1" />
                </div>
              ))}
            </div>
          ) : recentSessions.length === 0 && !isActive ? (
            <div className="flex flex-col items-center justify-center px-5 py-14 text-center sm:py-16">
              <div className="mb-4 flex size-16 items-center justify-center rounded-full bg-primary/8 sm:size-20">
                <Mic className="size-8 text-primary/40 sm:size-10" strokeWidth={1.2} />
              </div>
              <p className="text-sm font-medium text-content-muted">{t`No sessions yet`}</p>
              <p className="mt-1 text-xs text-content-subtle">
                {t`Press`} <kbd className="kbd kbd-xs">{t`Ctrl`}</kbd> + <kbd className="kbd kbd-xs">{t`Space`}</kbd> {t`to start dictating`}
              </p>
            </div>
          ) : recentSessions.length === 0 && isActive ? null : (
            <div className="divide-y divide-surface-border">
              {recentSessions.map((session) => {
                const time = new Date(session.startedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
                return (
                  <SessionDetailModal key={session.id} session={session}>
                    {(open) => (
                      <article className="block px-5 py-4 cursor-pointer transition-colors hover:bg-surface-hover" onClick={open}>
                        <div className="pt-1 text-xs tabular-nums text-content-subtle">{time}</div>
                        <div className="min-w-0">
                          <p className="text-sm leading-6 break-words text-content">
                            {session.transcript || t`Empty session`}
                          </p>
                          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-content-muted">
                            <span className="badge badge-ghost badge-sm">{session.words} {t`words`}</span>
                            <span>{session.seconds}{t`s`}</span>
                          </div>
                        </div>
                      </article>
                    )}
                  </SessionDetailModal>
                );
              })}
            </div>
          )}
        </div>
      </section>
    </div>
    </ModelSetup>
  );
}
