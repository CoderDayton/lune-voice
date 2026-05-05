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
import { useEffect, useId, useRef, useState } from 'react';
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

export function SessionDetailModal({ session, children }: SessionDetailModalProps) {
  const { t } = useLingui();
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [copied, setCopied] = useState(false);
  const titleId = useId();
  const transcriptId = useId();

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

  // Ctrl+C copies full transcript when modal is open and no text is selected
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'c' && (e.ctrlKey || e.metaKey) && !window.getSelection()?.toString()) {
        e.preventDefault();
        handleCopy();
      }
    }
    dialog.addEventListener('keydown', onKeyDown);
    return () => dialog.removeEventListener('keydown', onKeyDown);
  });

  return (
    <>
      {children(() => dialogRef.current?.showModal())}

      <dialog
        ref={dialogRef}
        className="modal backdrop-blur-sm"
        aria-labelledby={titleId}
        aria-describedby={transcriptId}
      >
        <div className="modal-box max-w-lg p-0 overflow-hidden border border-surface-border sm:max-w-xl">
          {/* Header */}
          <div className="flex items-center justify-between px-5 pt-5 pb-3 sm:px-6 sm:pt-6">
            <div>
              <h3 id={titleId} className="text-base font-semibold tracking-tight">
                {date}
              </h3>
              <p className="text-sm text-content-subtle mt-0.5 tabular-nums">
                {time} · {relative}
              </p>
            </div>
            <div className="flex items-center gap-1">
              {session.transcript && (
                <>
                  <button
                    type="button"
                    onClick={handleDownload}
                    className="btn btn-ghost btn-sm btn-circle"
                    aria-label={t`Download transcript`}
                  >
                    <Download size={15} strokeWidth={1.8} />
                  </button>
                  <button
                    type="button"
                    onClick={handleCopy}
                    className="btn btn-ghost btn-sm btn-circle"
                    aria-label={t`Copy transcript`}
                  >
                    {copied ? (
                      <Check size={15} strokeWidth={2.5} className="text-success" />
                    ) : (
                      <Copy size={15} strokeWidth={1.8} />
                    )}
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Transcript */}
          <div className="px-5 pb-4 sm:px-6">
            {session.transcript ? (
              <div className="rounded-lg border border-surface-border bg-surface-inset p-5">
                <p
                  id={transcriptId}
                  className="text-base leading-relaxed text-content whitespace-pre-wrap break-words sm:text-lg"
                >
                  {session.transcript}
                </p>
              </div>
            ) : (
              <p
                id={transcriptId}
                className="text-sm italic text-content-subtle"
              >{t`No transcript`}</p>
            )}
          </div>

          {/* Metadata grid */}
          <div className="border-t border-surface-border bg-surface-inset px-5 py-4 sm:px-6">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetaItem icon={Hash} label={t`Words`} value={String(session.words)} />
              <MetaItem icon={Clock} label={t`Duration`} value={formatDuration(session.seconds)} />
              <MetaItem icon={Mic} label={t`Engine`} value={formatBackend(session.backend)} />
              <MetaItem icon={Palette} label={t`Style`} value={session.style} />
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

          {/* Actions */}
          <div className="modal-action px-5 pb-4 pt-0 sm:px-6 justify-between">
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
              <button
                type="submit"
                className="btn btn-ghost btn-sm text-content-muted"
              >{t`Close`}</button>
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
    <div className="flex items-center gap-3">
      <Icon size={14} strokeWidth={1.6} className="text-content-dim shrink-0" />
      <div className="min-w-0">
        <p className="text-[0.65rem] uppercase tracking-wider text-content-dim">{label}</p>
        <p className="text-sm font-medium truncate">{value}</p>
      </div>
    </div>
  );
}

function formatBackend(backend: string): string {
  if (backend.includes('Qwen3') || backend.includes('qwen3')) return 'Qwen3-ASR';
  if (backend.includes('OpenAi')) return 'Cloud';
  return backend;
}
