import { useLingui } from '@lingui/react/macro';
import { Link } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { Clock, Hash, MessageSquareText, Mic } from 'lucide-react';
import { InfoBadge, StatusBadge } from '../../components/ui/status-badge';
import {
  getSessionStats,
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from '../../store/workspace-store';
import { SessionDetailModal } from '../history/components/session-detail-modal';
import { ModelSetup } from './components/model-setup';

export function OverviewPage() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const dictationMode = useStore(workspaceStore, (s) => s.settings.dictationMode);
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const stats = useStore(workspaceStore, getSessionStats);
  const recentSessions = useStore(workspaceStore, (s) => s.historySessions.slice(0, 5));

  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnected =
    serviceStatus === 'ready' || serviceStatus === 'listening' || serviceStatus === 'processing';
  const isLoading = serviceStatus === 'Connecting' || loading;
  const showSkeleton = loading && stats.count === 0;

  return (
    <ModelSetup>
      <div className="page-stack">
        <section className="panel-surface">
          <div className="card-body gap-5 p-5 sm:p-6">
            <div className="flex flex-wrap items-center gap-3 text-base">
              <StatusBadge variant={isLoading ? 'warning' : isConnected ? 'success' : 'error'}>
                {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
              </StatusBadge>
              <InfoBadge>{summarizeAsrBackend(asrBackend)}</InfoBadge>
              <InfoBadge>{summarizeDictationMode(dictationMode)}</InfoBadge>
            </div>

            {showSkeleton ? (
              <div className="grid gap-3 sm:grid-cols-3">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="metric-card">
                    <span className="skeleton-shimmer h-3 w-12" />
                    <span className="skeleton-shimmer mt-3 h-8 w-20" />
                  </div>
                ))}
              </div>
            ) : stats.count > 0 ? (
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="metric-card interactive-card">
                  <div className="mb-4 flex items-center justify-between text-content-subtle">
                    <span className="section-label">{t`Words`}</span>
                    <Hash size={20} strokeWidth={1.5} className="text-primary/55" />
                  </div>
                  <div className="text-3xl font-semibold tracking-tight tabular-nums">
                    {stats.totalWords.toLocaleString()}
                  </div>
                </div>
                <div className="metric-card interactive-card">
                  <div className="mb-4 flex items-center justify-between text-content-subtle">
                    <span className="section-label">{t`Minutes`}</span>
                    <Clock size={20} strokeWidth={1.5} className="text-secondary/55" />
                  </div>
                  <div className="text-3xl font-semibold tracking-tight tabular-nums">
                    {stats.totalMinutes}
                  </div>
                </div>
                <div className="metric-card interactive-card">
                  <div className="mb-4 flex items-center justify-between text-content-subtle">
                    <span className="section-label">{t`Sessions`}</span>
                    <MessageSquareText size={20} strokeWidth={1.5} className="text-accent/60" />
                  </div>
                  <div className="text-3xl font-semibold tracking-tight tabular-nums">
                    {stats.count}
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-box border border-dashed border-surface-border bg-surface-inset px-5 py-6 text-sm text-content-muted">
                {t`Your session summary will appear here once you start dictating.`}
              </div>
            )}
          </div>
        </section>

        <section className="panel-surface overflow-hidden">
          <div className="card-body p-0">
            <div className="flex items-center justify-between px-5 pb-3 pt-5">
              <div>
                <h2 className="section-label">{t`Recent`}</h2>
                <p className="mt-1 text-sm text-content-subtle tabular-nums">
                  {stats.count} {stats.count === 1 ? t`session` : t`sessions`}
                </p>
              </div>
              {stats.count > 5 && (
                <Link to="/history" className="btn btn-soft btn-primary btn-sm normal-case">
                  {t`View all`}
                </Link>
              )}
            </div>
            {showSkeleton ? (
              <div className="divide-y divide-surface-border">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="px-5 py-4 flex flex-col gap-3">
                    <span className="skeleton-shimmer h-3 w-16" />
                    <span className="skeleton-shimmer h-4 w-full" />
                    <span className="skeleton-shimmer h-4 w-3/4" />
                    <span className="skeleton-shimmer h-3 w-24 mt-1" />
                  </div>
                ))}
              </div>
            ) : recentSessions.length === 0 && !isActive ? (
              <div className="flex flex-col items-center justify-center px-5 py-14 text-center sm:py-16">
                <div className="token-icon mb-4 size-16 rounded-full sm:size-20">
                  <Mic className="size-9 sm:size-11" strokeWidth={1.2} />
                </div>
                <p className="text-sm font-medium text-content-muted">{t`No sessions yet`}</p>
                <p className="mt-1 text-sm text-content-subtle">
                  {t`Press`} <kbd className="kbd kbd-xs">{hotkeyToggle || t`your hotkey`}</kbd>{' '}
                  {t`to start dictating`}
                </p>
              </div>
            ) : recentSessions.length === 0 && isActive ? null : (
              <div className="divide-y divide-surface-border">
                {recentSessions.map((session) => {
                  const time = new Date(session.startedAt).toLocaleTimeString([], {
                    hour: 'numeric',
                    minute: '2-digit',
                  });
                  return (
                    <SessionDetailModal key={session.id} session={session}>
                      {(open) => (
                        <button
                          type="button"
                          className="interactive-row block w-full text-left px-5 py-4 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-primary"
                          onClick={open}
                          aria-haspopup="dialog"
                          aria-label={t`Open session from ${time}`}
                        >
                          <div className="pt-1 text-sm tabular-nums text-content-subtle">
                            {time}
                          </div>
                          <div className="min-w-0">
                            <p className="text-base leading-6 break-words text-content">
                              {session.transcript || t`Empty session`}
                            </p>
                            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-content-muted">
                              <span className="badge badge-soft badge-sm">
                                {session.words} {t`words`}
                              </span>
                              <span>
                                {session.seconds}
                                {t`s`}
                              </span>
                            </div>
                          </div>
                        </button>
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
