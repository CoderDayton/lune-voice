import { useLingui } from '@lingui/react/macro';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import {
  Check,
  CircleCheck,
  CircleX,
  Clock,
  Copy,
  Download,
  Hash,
  Mic,
  Palette,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ConfirmDialog } from '../../../components/feedback/confirm-dialog';
import { type HistorySession, removeHistorySession } from '../../../store/workspace-store';

interface SessionDetailModalProps {
  session: HistorySession;
  children: (open: () => void) => React.ReactNode;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Presentational detail for a single session — header (date + copy/download),
 * transcript, and metadata. Shared by the mobile modal and the desktop
 * master-detail pane. Owns copy/download and the Ctrl+C shortcut, scoped to
 * its own root so it works in either host.
 */
export function SessionDetailContent({ session }: { session: HistorySession }) {
  const { t } = useLingui();
  const [copied, setCopied] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const startDate = new Date(session.startedAt);
  const date = startDate.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
  const time = startDate.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const relative = relativeTime(session.startedAt);

  const isCompleted = session.status === 'Completed';
  const isFailed = session.status.startsWith('Failed');

  async function handleDownload() {
    if (!session.transcript) return;
    const d = new Date(session.startedAt);
    const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
    try {
      const path = await save({
        title: t`Save Transcript`,
        defaultPath: `lune-${stamp}.txt`,
        filters: [{ name: 'Text', extensions: ['txt'] }],
      });
      if (path) await writeTextFile(path, session.transcript);
    } catch (e) {
      console.warn('[transcript-download]', e);
    }
  }

  function handleCopy() {
    if (!session.transcript) return;
    navigator.clipboard
      .writeText(session.transcript)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  }

  // Ctrl/Cmd+C copies the full transcript when focus is inside this detail and
  // the user has not made a manual text selection.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'c' && (e.ctrlKey || e.metaKey) && !window.getSelection()?.toString()) {
        e.preventDefault();
        handleCopy();
      }
    }
    el.addEventListener('keydown', onKeyDown);
    return () => el.removeEventListener('keydown', onKeyDown);
  });

  return (
    <div ref={rootRef}>
      {/* Header */}
      <div className="flex items-start justify-between gap-4 border-b border-surface-border px-4 py-4 sm:px-5">
        <div className="min-w-0">
          <h3 className="truncate text-base font-semibold tracking-tight text-content-strong">
            {date}
          </h3>
          <p className="mt-1 text-sm tabular-nums text-content-subtle">
            {time} · {relative}
          </p>
        </div>
        {session.transcript && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={handleDownload}
              className="btn btn-soft btn-sm btn-circle"
              aria-label={t`Download transcript`}
            >
              <Download size={15} strokeWidth={1.8} />
            </button>
            <button
              type="button"
              onClick={handleCopy}
              className="btn btn-soft btn-sm btn-circle"
              aria-label={t`Copy transcript`}
            >
              {copied ? (
                <Check size={15} strokeWidth={2.5} className="text-success" />
              ) : (
                <Copy size={15} strokeWidth={1.8} />
              )}
            </button>
          </div>
        )}
      </div>

      {/* Transcript */}
      <div className="px-4 py-4 sm:px-5">
        {session.transcript ? (
          <div className="rounded-field border border-surface-border bg-surface-inset p-4">
            <p className="whitespace-pre-wrap break-words text-[1rem] leading-7 text-content">
              {session.transcript}
            </p>
          </div>
        ) : (
          <p className="text-sm italic text-content-subtle">{t`No transcript`}</p>
        )}
      </div>

      {/* Metadata */}
      <div className="border-t border-surface-border bg-surface-inset/70 px-4 py-4 sm:px-5">
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          <MetaItem icon={Hash} label={t`Words`} value={String(session.words)} />
          <MetaItem icon={Clock} label={t`Duration`} value={formatDuration(session.seconds)} />
          <MetaItem icon={Mic} label={t`Engine`} value={formatBackend(session.backend)} />
          <MetaItem icon={Palette} label={t`Style`} value={session.style || t`None`} />
        </div>
        <div className="mt-3 flex items-center gap-3 text-sm text-content-subtle">
          {isCompleted ? (
            <>
              <CircleCheck size={13} strokeWidth={2} className="text-success/60" />{' '}
              <span>{t`Completed`}</span>
            </>
          ) : isFailed ? (
            <>
              <CircleX size={13} strokeWidth={2} className="text-error/60" />{' '}
              <span>{session.status}</span>
            </>
          ) : (
            <span>{session.status}</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function SessionDetailModal({ session, children }: SessionDetailModalProps) {
  const { t } = useLingui();
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  return (
    <>
      {children(() => dialogRef.current?.showModal())}

      <dialog ref={dialogRef} className="modal backdrop-blur-sm" aria-label={t`Session details`}>
        <div className="modal-box max-w-lg overflow-hidden border border-surface-border p-0 sm:max-w-xl">
          <SessionDetailContent session={session} />

          {/* Actions */}
          <div className="modal-action justify-between border-t border-surface-border px-5 pb-4 pt-3 sm:px-6">
            <ConfirmDialog
              title={t`Delete session?`}
              description={t`This will permanently remove this dictation session and its transcript.`}
              triggerLabel={t`Delete`}
              onConfirm={() => {
                dialogRef.current?.close();
                removeHistorySession(session.id);
              }}
            />
            <form method="dialog">
              <button type="submit" className="btn btn-soft btn-sm">{t`Close`}</button>
            </form>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button type="submit">close</button>
        </form>
      </dialog>
    </>
  );
}

function MetaItem({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Clock;
  label: string;
  value: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-field border border-surface-border/70 bg-surface-elevated px-3 py-2">
      <Icon size={14} strokeWidth={1.6} className="shrink-0 text-content-subtle" />
      <div className="min-w-0">
        <p className="text-[0.68rem] uppercase tracking-[0.12em] text-content-subtle">{label}</p>
        <p className="truncate text-sm font-medium text-content">{value}</p>
      </div>
    </div>
  );
}

function formatBackend(backend: string): string {
  if (backend.includes('WhisperCpp')) return 'Whisper';
  if (backend.includes('OpenAi')) return 'Cloud';
  return backend;
}
