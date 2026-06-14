import { useLingui } from '@lingui/react/macro';
import { Link } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import { Activity, CalendarDays, Clock, MessageSquareText, Mic } from 'lucide-react';
import { InfoBadge, StatusBadge } from '../../components/ui/status-badge';
import {
  getSessionStats,
  getTodayStats,
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from '../../store/workspace-store';
import { SessionDetailModal } from '../history/components/session-detail-modal';
import { useCountUp } from './use-count-up';

const AUDIO_BARS = 7;

export function OverviewPage() {
  const { t } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);
  const audioLevel = useStore(workspaceStore, (s) => s.runtime.audioLevel);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const dictationMode = useStore(workspaceStore, (s) => s.settings.dictationMode);
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);
  const stats = useStore(workspaceStore, getSessionStats);
  const today = useStore(workspaceStore, getTodayStats);
  const historySessions = useStore(workspaceStore, (s) => s.historySessions);
  const recentSessions = historySessions.slice(0, 5);

  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnected =
    serviceStatus === 'ready' || serviceStatus === 'listening' || serviceStatus === 'processing';
  const isLoading = serviceStatus === 'Connecting' || loading;
  const showSkeleton = loading && stats.count === 0;

  // Animated counters — restyle of data that already streams in; honors
  // prefers-reduced-motion inside the hook.
  const wordsCount = useCountUp(stats.totalWords);
  const minutesCount = useCountUp(stats.totalMinutes);
  const sessionsCount = useCountUp(stats.count);
  const todayWords = useCountUp(today.words);
  const wpm = stats.totalMinutes > 0 ? Math.round(stats.totalWords / stats.totalMinutes) : 0;
  const avgWordsPerSession = stats.count > 0 ? Math.round(stats.totalWords / stats.count) : 0;
  const wpmCount = useCountUp(wpm);
  // Today's speaking pace vs the all-time average — a quick trending-up/down
  // read on the hero card, mirroring the all-time/today split on the left.
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const todayStartMs = todayStart.getTime();
  let wordsToday = 0;
  let secondsToday = 0;
  for (const s of historySessions) {
    if (new Date(s.startedAt).getTime() >= todayStartMs) {
      wordsToday += s.words;
      secondsToday += s.seconds;
    }
  }
  const wpmToday = secondsToday > 0 ? Math.round(wordsToday / (secondsToday / 60)) : null;
  const wpmDelta = wpmToday !== null && wpm > 0 ? wpmToday - wpm : null;

  const hour = new Date().getHours();
  const greeting = hour < 12 ? t`Good morning` : hour < 18 ? t`Good afternoon` : t`Good evening`;

  const statusTone = isLoading ? 'warning' : isConnected ? 'success' : 'error';
  const dotClass =
    statusTone === 'success' ? 'bg-success' : statusTone === 'warning' ? 'bg-warning' : 'bg-error';

  if (showSkeleton) {
    return (
      <div className="bento-board">
        <div className="bento-cell col-span-12 p-space-lg md:col-span-6 md:row-span-2">
          <span className="skeleton-shimmer h-3 w-14" />
          <span className="skeleton-shimmer mt-auto h-12 w-40" />
        </div>
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="bento-cell-elevated col-span-12 gap-3 p-space-md md:col-span-3">
            <span className="skeleton-shimmer h-3 w-12" />
            <span className="skeleton-shimmer mt-auto h-7 w-16" />
          </div>
        ))}
        <div className="bento-cell col-span-12 gap-3 p-space-md">
          <span className="skeleton-shimmer h-3 w-16" />
          <span className="skeleton-shimmer h-4 w-full" />
          <span className="skeleton-shimmer h-4 w-3/4" />
        </div>
      </div>
    );
  }

  return (
    <div className="bento-board">
      {/* Brand hero — slim identity row; live status folds into its right side */}
      <div className="col-span-12 flex flex-col gap-3 px-1 pt-1 pb-2 select-none sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <h1 className="font-display text-4xl font-semibold tracking-display sm:text-5xl">
            <span className="text-content-strong">Lune</span>{' '}
            <span className="text-primary">Voice</span>
          </h1>
          <p className="mt-1.5 text-sm text-content-subtle">
            {greeting} — {t`hold the hotkey and speak, your words land where your cursor is.`}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2 pb-1">
          <StatusBadge variant={statusTone}>
            {isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`}
          </StatusBadge>
          <InfoBadge>{summarizeAsrBackend(asrBackend)}</InfoBadge>
          <InfoBadge>
            {t`Output`}: {summarizeDictationMode(dictationMode)}
          </InfoBadge>
        </div>
      </div>

      {/* Hero metric — Words dominates the board */}
      <div className="bento-cell interactive-card col-span-12 p-space-lg md:col-span-6 md:row-span-2">
        <div className="flex items-center justify-between text-content-subtle">
          <span className="section-label">{t`Total Words`}</span>
          <span className="section-label">{t`Words Per Minute`}</span>
        </div>
        <div className="mt-auto flex items-end justify-between gap-4 pt-6">
          <div className="min-w-0">
            <div className="font-display text-display-2xl font-semibold tracking-display tabular-nums text-content-strong">
              {wordsCount.toLocaleString()}
            </div>
            <p className="mt-2 text-sm text-content-muted tabular-nums">
              {t`all-time`}
              {` · ${today.words > 0 ? '↑ ' : ''}${todayWords.toLocaleString()} ${t`today`}`}
            </p>
          </div>
          <div
            aria-hidden="true"
            className="w-px self-stretch bg-linear-to-b from-transparent via-surface-border-strong to-transparent"
          />
          <div className="shrink-0 text-right">
            <div className="font-display text-display-2xl font-semibold tracking-display tabular-nums text-content-strong">
              {wpmCount.toLocaleString()}
            </div>
            <p className="mt-2 text-sm text-content-muted tabular-nums">
              {t`all-time avg`}
              {` · ${wpmDelta && wpmDelta > 0 ? '↑ ' : wpmDelta && wpmDelta < 0 ? '↓ ' : ''}${Math.abs(wpmDelta ?? 0)} ${t`today`}`}
            </p>
          </div>
        </div>
      </div>

      {/* Stat cells — tighter density, asymmetric to the hero */}
      <div className="bento-cell-elevated interactive-card col-span-12 gap-4 p-space-md md:col-span-3">
        <div className="flex items-center justify-between text-content-subtle">
          <span className="section-label">{t`Speaking Time`}</span>
          <Clock size={20} strokeWidth={1.5} className="text-secondary/55" />
        </div>
        <div className="mt-auto">
          <div className="text-3xl font-semibold tracking-tight tabular-nums text-content-strong">
            {minutesCount.toLocaleString()}
          </div>
          <p className="mt-1 text-sm text-content-muted tabular-nums">
            {stats.totalMinutes === 1 ? t`minute` : t`minutes`}
          </p>
        </div>
      </div>

      <div className="bento-cell-elevated interactive-card col-span-12 gap-4 p-space-md md:col-span-3">
        <div className="flex items-center justify-between text-content-subtle">
          <span className="section-label">{t`Sessions`}</span>
          <MessageSquareText size={20} strokeWidth={1.5} className="text-accent/60" />
        </div>
        <div className="mt-auto">
          <div className="text-3xl font-semibold tracking-tight tabular-nums text-content-strong">
            {sessionsCount.toLocaleString()}
          </div>
          <p className="mt-1 text-sm text-content-muted tabular-nums">
            {avgWordsPerSession > 0 ? t`~${avgWordsPerSession} words each` : t`per session`}
          </p>
        </div>
      </div>

      <div className="bento-cell-elevated col-span-12 gap-4 p-space-md md:col-span-3">
        <div className="flex items-center justify-between text-content-subtle">
          <span className="section-label">{t`Today`}</span>
          <CalendarDays size={20} strokeWidth={1.5} className="text-primary/55" />
        </div>
        <div className="mt-auto">
          <div className="text-3xl font-semibold tracking-tight tabular-nums text-content-strong">
            {today.count.toLocaleString()}
          </div>
          <p className="mt-1 text-sm text-content-muted tabular-nums">
            {today.count === 1 ? t`session` : t`sessions`}
          </p>
        </div>
      </div>

      {/* Live audio meter — bound to runtime.audioLevel */}
      <div className="bento-cell-elevated col-span-12 gap-3 p-space-md md:col-span-3">
        <div className="flex items-center justify-between text-content-subtle">
          <span className="section-label">{t`Audio`}</span>
          <Activity size={20} strokeWidth={1.5} className="text-accent/60" />
        </div>
        <div className="audio-meter mt-auto" role="img" aria-label={t`Microphone input level`}>
          {Array.from({ length: AUDIO_BARS }).map((_, i) => {
            const on = isActive && audioLevel >= (i + 0.5) / AUDIO_BARS;
            const height = 28 + (i / (AUDIO_BARS - 1)) * 72;
            return (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed-length meter, index is the identity
                key={i}
                className="audio-meter-bar"
                data-on={on}
                style={{ height: `${height}%` }}
              />
            );
          })}
        </div>
        <p className="flex items-center gap-2 text-sm text-content-muted">
          <span className="relative inline-flex size-2 items-center justify-center">
            {isActive && (
              <span className={`status-pulse absolute inset-0 rounded-full ${dotClass}`} />
            )}
            <span className={`relative size-2 rounded-full ${dotClass}`} />
          </span>
          {isActive ? t`Listening…` : isConnected ? t`Idle` : t`Offline`}
        </p>
      </div>

      {/* Recent feed — dense rows */}
      <section className="bento-cell col-span-12 overflow-hidden">
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
        {recentSessions.length === 0 && !isActive ? (
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
                      <div className="pt-1 text-sm tabular-nums text-content-subtle">{time}</div>
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
      </section>
    </div>
  );
}
