// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Link } from '@tanstack/react-router';
import { useStore } from '@tanstack/react-store';
import {
  Activity,
  CalendarDays,
  ChevronRight,
  Clock,
  Gauge,
  HardDrive,
  Mic,
  PlugZap,
  Search,
} from 'lucide-react';
import { LIMITS } from '../../lib/backend/generated/limits';
import {
  getSessionStats,
  getTodayStats,
  type HistorySession,
  loadWorkspaceSnapshot,
  summarizeAsrBackend,
  summarizeDictationMode,
  workspaceStore,
} from '../../store/workspace-store';
import { SessionDetailModal } from '../history/components/session-detail-modal';
import { useCountUp } from './use-count-up';

const AUDIO_BARS = 7;
// The pane shows a fixed slice of the recent feed; the full list lives in
// History. Enough rows to fill a tall window — the list scrolls inside the
// pane when the window is shorter than that.
const RECENT_LIMIT = 12;

// The only subscriber to `runtime.audioLevel`, which updates on every raw
// audio-level stream event with no coalescing. Subscribing from OverviewPage
// itself re-rendered the whole pane — animated counters, session list and
// all — at the raw event rate for the sake of seven bars.
function AudioMeter({ isActive, label }: { isActive: boolean; label: string }) {
  const audioLevel = useStore(workspaceStore, (s) => s.runtime.audioLevel);
  const bars = Array.from({ length: AUDIO_BARS }, (_, i) => ({
    id: i,
    on: isActive && audioLevel >= (i + 0.5) / AUDIO_BARS,
    height: 30 + (i / (AUDIO_BARS - 1)) * 70,
  }));
  return (
    <div className="audio-meter" role="img" aria-label={label}>
      {bars.map((bar) => (
        <span
          key={bar.id}
          className="audio-meter-bar"
          data-on={bar.on}
          style={{ height: `${bar.height}%` }}
        />
      ))}
    </div>
  );
}

interface StatStripProps {
  isActive: boolean;
  isConnected: boolean;
  /** Runtime unreachable — the figures shown are the last known values. */
  stale: boolean;
}

/** The hairline-divided read-out band at the top of the pane. */
function StatStrip({ isActive, isConnected, stale }: StatStripProps) {
  const { t } = useLingui();
  const stats = useStore(workspaceStore, getSessionStats);
  const today = useStore(workspaceStore, getTodayStats);

  // Animated counters — restyle of data that already streams in; honors
  // prefers-reduced-motion inside the hook.
  const wordsCount = useCountUp(stats.totalWords);
  const minutesCount = useCountUp(stats.totalMinutes);
  const todayWords = useCountUp(today.words);
  const wpm = stats.totalMinutes > 0 ? Math.round(stats.totalWords / stats.totalMinutes) : 0;
  const wpmCount = useCountUp(wpm);
  // Today's speaking pace against the all-time average — a trending-up/down
  // read that mirrors the all-time/today split on the hero metric.
  const wpmToday = today.seconds > 0 ? Math.round(today.words / (today.seconds / 60)) : null;
  const wpmDelta = wpmToday !== null && wpm > 0 ? wpmToday - wpm : null;
  const todayMinutes = Math.round(today.seconds / 60);

  return (
    <div className="ov-stats" data-stale={stale}>
      <div className="ov-stat">
        <div className="ov-label">{t`Total words`}</div>
        <div className="ov-value">
          <div className="flex min-w-0 flex-wrap items-baseline gap-2.5">
            <span className="ov-metric-hero">{wordsCount.toLocaleString()}</span>
            {/* Hidden when today is everything: "↑ 120" beside "120" says nothing. */}
            {today.words > 0 && today.words < stats.totalWords && (
              <span className="text-sm tabular-nums text-success">
                ↑ {todayWords.toLocaleString()}
              </span>
            )}
          </div>
        </div>
        <p className="ov-note">{t`all-time · today`}</p>
      </div>

      <div className="ov-stat">
        <div className="ov-label">
          {t`Speaking time`}
          <Clock size={15} strokeWidth={1.6} className="text-secondary/60" />
        </div>
        <div className="ov-value">
          <span className="ov-metric">
            {minutesCount.toLocaleString()}
            <span className="text-base font-normal text-content-subtle"> {t`min`}</span>
          </span>
        </div>
        <p className="ov-note tabular-nums">{t`${todayMinutes} min today`}</p>
      </div>

      <div className="ov-stat">
        <div className="ov-label">
          {t`WPM`}
          <Gauge size={15} strokeWidth={1.6} className="text-accent/65" />
        </div>
        <div className="ov-value">
          <span className="ov-metric">{wpmCount.toLocaleString()}</span>
        </div>
        <p className="ov-note tabular-nums">
          {wpmDelta !== null && wpmDelta !== 0 && (
            <span className={wpmDelta > 0 ? 'text-success' : 'text-warning'}>
              {wpmDelta > 0 ? '↑' : '↓'} {Math.abs(wpmDelta)}{' '}
            </span>
          )}
          {t`vs all-time avg`}
        </p>
      </div>

      <div className="ov-stat">
        <div className="ov-label">
          {t`Today`}
          <CalendarDays size={15} strokeWidth={1.6} className="text-primary/60" />
        </div>
        <div className="ov-value">
          <span className="ov-metric">{today.count.toLocaleString()}</span>
        </div>
        <p className="ov-note">{today.count === 1 ? t`session` : t`sessions`}</p>
      </div>

      <div className="ov-stat">
        <div className="ov-label">
          {t`Audio`}
          <Activity size={15} strokeWidth={1.6} className="text-accent/65" />
        </div>
        <div className="ov-value">
          <AudioMeter isActive={isActive} label={t`Microphone input level`} />
        </div>
        <p className="ov-note">
          {isActive ? t`Listening…` : isConnected ? t`Idle · mic ready` : t`Offline`}
        </p>
      </div>
    </div>
  );
}

/** One tabular session row; opens the shared detail modal on click. */
function SessionRow({ session }: { session: HistorySession }) {
  const { t } = useLingui();
  const time = new Date(session.startedAt).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
  return (
    <SessionDetailModal session={session}>
      {(open) => (
        <button
          type="button"
          className="ov-row ov-row-session"
          onClick={open}
          aria-haspopup="dialog"
          aria-label={t`Open session from ${time}`}
        >
          <span className="font-mono text-sm tabular-nums text-content-subtle">{time}</span>
          <span className="truncate text-base text-content">
            {session.transcript || t`Empty session`}
          </span>
          <span className="ov-num text-right text-sm tabular-nums text-content-muted">
            {session.words}
          </span>
          <span className="ov-num text-right text-sm tabular-nums text-content-subtle">
            {session.seconds}
            {t`s`}
          </span>
        </button>
      )}
    </SessionDetailModal>
  );
}

function SessionList({ sessions }: { sessions: HistorySession[] }) {
  const { t } = useLingui();
  const hotkeyToggle = useStore(workspaceStore, (s) => s.settings.hotkeyToggle);

  if (sessions.length === 0) {
    return (
      <div className="ov-list flex flex-col items-center justify-center px-5 py-12 text-center">
        <div className="token-icon size-13 rounded-full">
          <Mic className="size-6" strokeWidth={1.4} />
        </div>
        <p className="mt-3.5 text-base font-medium text-content-muted">{t`No sessions yet`}</p>
        <p className="ov-note mt-1.5">
          {t`Press`} <kbd className="kbd kbd-xs">{hotkeyToggle || t`your hotkey`}</kbd>{' '}
          {t`to start dictating`}
        </p>
      </div>
    );
  }

  return (
    <div className="ov-table">
      <div className="ov-row ov-row-head">
        <span>{t`Time`}</span>
        <span>{t`Transcript`}</span>
        <span className="ov-num text-right">{t`Words`}</span>
        <span className="ov-num text-right">{t`Length`}</span>
      </div>
      <div className="ov-list">
        {sessions.map((session) => (
          <SessionRow key={session.id} session={session} />
        ))}
      </div>
    </div>
  );
}

function OverviewSkeleton() {
  const cells = [0, 1, 2, 3, 4];
  return (
    <div className="ov-page">
      <div className="flex shrink-0 items-center justify-between gap-4 px-1 py-2">
        <span className="skeleton-shimmer h-4 w-96 max-w-full" />
        <span className="skeleton-shimmer hidden h-3 w-52 sm:block" />
      </div>
      <div className="ov-panel">
        <div className="ov-stats">
          {cells.map((i) => (
            <div key={i} className="ov-stat">
              <span className="skeleton-shimmer h-3 w-20" />
              <span className="skeleton-shimmer h-8 w-24 self-center" />
              <span className="skeleton-shimmer h-3 w-16" />
            </div>
          ))}
        </div>
        <div className="ov-bar">
          <span className="skeleton-shimmer h-2.5 w-32" />
        </div>
        <div className="ov-table ov-list">
          {cells.map((i) => (
            <div key={i} className="ov-row py-4">
              <span className="skeleton-shimmer h-3 w-14" />
              <span className="skeleton-shimmer h-3.5 w-full" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function OverviewPage() {
  const { t, i18n } = useLingui();
  const serviceStatus = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);
  const asrBackend = useStore(workspaceStore, (s) => s.settings.asrBackend);
  const dictationMode = useStore(workspaceStore, (s) => s.settings.dictationMode);
  const historyDays = useStore(workspaceStore, (s) => s.settings.sessionHistoryDays);
  const totalSessions = useStore(workspaceStore, (s) => s.sessionTotals.sessions);
  const historySessions = useStore(workspaceStore, (s) => s.historySessions);
  const recentSessions = historySessions.slice(0, RECENT_LIMIT);

  const isActive = serviceStatus === 'listening' || serviceStatus === 'processing';
  const isConnected = serviceStatus === 'ready' || isActive;
  const isLoading = serviceStatus === 'Connecting' || loading;

  if (loading && totalSessions === 0) return <OverviewSkeleton />;

  const hour = new Date().getHours();
  const greeting = hour < 12 ? t`Good morning.` : hour < 18 ? t`Good afternoon.` : t`Good evening.`;
  const dotClass = isLoading ? 'bg-warning' : isConnected ? 'bg-success' : 'bg-error';
  const statusLabel = isLoading ? t`Connecting` : isConnected ? t`Ready` : t`Disconnected`;

  const isStale = !isConnected && !isLoading;

  return (
    <div className="ov-page">
      {/* Greeting line — the pane below is the page, so this stays a single
          quiet row rather than a hero block. */}
      <div className="flex shrink-0 flex-col gap-2 px-1 py-2 select-none sm:flex-row sm:items-center sm:justify-between">
        <p className="min-w-0 text-base text-content-muted">
          <span className="font-medium text-content-strong">{greeting}</span>{' '}
          {t`Hold the hotkey and speak, and your words land wherever your cursor is.`}
        </p>
        <div className="ov-meta">
          <span className="inline-flex items-center gap-1.5 text-content">
            <span className="relative inline-flex size-2 items-center justify-center">
              {isActive && (
                <span className={`status-pulse absolute inset-0 rounded-full ${dotClass}`} />
              )}
              <span className={`relative size-2 rounded-full ${dotClass}`} />
            </span>
            {statusLabel}
          </span>
          <span className="ov-sep" aria-hidden="true" />
          <span>{summarizeAsrBackend(i18n, asrBackend)}</span>
          <span className="ov-sep" aria-hidden="true" />
          <span>{summarizeDictationMode(i18n, dictationMode)}</span>
        </div>
      </div>

      <section className="ov-panel">
        {isStale && (
          <div className="ov-alert">
            <PlugZap size={17} strokeWidth={1.7} className="shrink-0 text-error" />
            <span className="min-w-0 grow">
              {t`The local runtime isn't responding. The figures below are the last known values.`}
            </span>
            <button
              type="button"
              className="btn btn-soft btn-sm"
              onClick={() => void loadWorkspaceSnapshot()}
            >
              {t`Retry`}
            </button>
          </div>
        )}

        <StatStrip isActive={isActive} isConnected={isConnected} stale={isStale} />

        <div className="ov-bar">
          <div className="flex items-baseline gap-2.5">
            <h2 className="ov-eyebrow text-content-muted">{t`Recent sessions`}</h2>
            <span className="ov-note tabular-nums">{t`${totalSessions} total`}</span>
          </div>
          {totalSessions > recentSessions.length && (
            <Link to="/history" className="ov-link text-sm">
              {t`View all`}
              <ChevronRight size={13} strokeWidth={2} />
            </Link>
          )}
        </div>

        <SessionList sessions={recentSessions} />

        <div className="ov-foot">
          {totalSessions > 0 && (
            <>
              <span className="tabular-nums">
                {t`Showing ${recentSessions.length} of ${totalSessions} sessions`}
              </span>
              <span className="ov-sep" aria-hidden="true" />
            </>
          )}
          <span className="tabular-nums">
            {historyDays == null
              ? t`History isn't saved`
              : historyDays === LIMITS.historyDaysForever
                ? t`History kept forever`
                : t`History kept ${historyDays} days`}
          </span>
          <span className="ov-sep" aria-hidden="true" />
          <span className="inline-flex items-center gap-1.5">
            <HardDrive size={13} strokeWidth={1.7} />
            {t`Stored on this device`}
          </span>
          <Link to="/history" className="ov-link ml-auto">
            <Search size={13} strokeWidth={1.8} />
            {t`Search sessions`}
          </Link>
        </div>
      </section>
    </div>
  );
}
