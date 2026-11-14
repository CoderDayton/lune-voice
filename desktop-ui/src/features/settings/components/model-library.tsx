// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { type RefObject, useEffect, useId, useRef, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import { formatBytes } from '../../../lib/format-bytes';
import { modelDownloads, useModelDownloads } from '../../model-downloads/model-downloads';
import { localizeText, parakeetModels, whisperModels } from '../data/settings-data';
import type { ModelStatusApi } from '../hooks/use-model-status';
import { ModelRow, type RowState } from './model-row';

type LocalModel = (typeof whisperModels)[number] | (typeof parakeetModels)[number];

type FamilyId = 'whisper' | 'parakeet';

const FAMILIES: { id: FamilyId; models: readonly LocalModel[] }[] = [
  { id: 'whisper', models: whisperModels },
  { id: 'parakeet', models: parakeetModels },
];

interface DeleteTarget {
  localName: string;
  name: string;
  inUse: boolean;
}

/**
 * A download the last window lifetime left running is still the user's: pick
 * it up once the library is known, so its progress shows here instead of
 * vanishing with the window.
 */
function useResumeDownloads(status: ModelStatusApi) {
  const resumed = useRef(false);
  useEffect(() => {
    if (resumed.current || status.loading) return;
    resumed.current = true;
    void backendClient
      .activeModelDownload()
      .catch(() => null)
      .then((active) =>
        modelDownloads.resume(
          status.models.map((m) => ({
            localName: m.localName,
            name: m.name,
            bytes: m.downloadBytes ?? 0,
            ready: m.ready,
          })),
          active,
        ),
      );
  }, [status.loading, status.models]);
}

/**
 * Every on-device model, one family per tab so the page keeps one height.
 * Each row downloads, shows its own progress, and deletes. Which installed
 * model is used is picked on the Transcription page.
 */
export function ModelLibrary({ status }: { status: ModelStatusApi }) {
  const { t, i18n } = useLingui();
  const run = useModelDownloads();
  const [picked, setPicked] = useState<FamilyId | null>(null);
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const panelId = useId();
  useResumeDownloads(status);

  const info = (m: LocalModel) => status.byName.get(m.localName);
  const isReady = (m: LocalModel) => info(m)?.ready ?? false;
  const inUse = (m: LocalModel) => (info(m)?.inUse ?? false) && isReady(m);
  const bytesOf = (m: LocalModel) => info(m)?.downloadBytes ?? 0;
  const label = (m: LocalModel) => localizeText(i18n, m.label);

  const inRun = (m: LocalModel) =>
    (run.status === 'running' || run.status === 'failed') &&
    run.models.some((r) => r.localName === m.localName);

  const stateOf = (m: LocalModel): RowState => {
    if (isReady(m)) return { kind: 'installed' };
    if (run.failed === m.localName && run.error) return { kind: 'failed', error: run.error };
    if (run.current === m.localName) {
      const total = run.models.find((r) => r.localName === m.localName)?.bytes || bytesOf(m);
      return { kind: 'downloading', done: run.currentBytes, total };
    }
    if (inRun(m)) return { kind: 'queued' };
    return { kind: 'available' };
  };

  const family = picked ?? FAMILIES.find((f) => f.models.some(inUse))?.id ?? FAMILIES[0].id;
  const shown = FAMILIES.find((f) => f.id === family) ?? FAMILIES[0];

  const installedBytes =
    status.diskBytes ??
    status.models.reduce((sum, m) => sum + (m.ready && m.sizeBytes != null ? m.sizeBytes : 0), 0);

  function download(m: LocalModel) {
    // A start outside a running run replaces it; carry over what a failure
    // left queued so those rows are not silently dropped.
    const queued =
      run.status === 'failed' ? run.models.filter((r) => run.queue.includes(r.localName)) : [];
    modelDownloads.start([
      ...queued,
      { localName: m.localName, name: info(m)?.name ?? label(m), bytes: bytesOf(m) },
    ]);
  }

  function closeDelete() {
    dialogRef.current?.close();
    setDeleting(null);
  }

  function confirmDelete() {
    if (deleting) void status.remove(deleting.localName);
    closeDelete();
  }

  const row = (m: LocalModel) => (
    <ModelRow
      key={m.value}
      label={label(m)}
      description={m.description === undefined ? '' : localizeText(i18n, m.description)}
      size={bytesOf(m) > 0 ? formatBytes(bytesOf(m)) : m.size}
      state={stateOf(m)}
      inUse={inUse(m)}
      onDownload={() => download(m)}
      onCancel={() => modelDownloads.drop(m.localName)}
      onRetry={() => modelDownloads.retry()}
      onDelete={() => {
        setDeleting({ localName: m.localName, name: label(m), inUse: inUse(m) });
        dialogRef.current?.showModal();
      }}
    />
  );

  const familyLabel = useFamilyLabels();
  const familyNote: Record<FamilyId, string> = {
    whisper: t`Nearly 100 languages. Bigger is more accurate, but slower.`,
    parakeet: t`25 European languages. Fast on a CPU.`,
  };

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <FamilyTabs
          selected={family}
          panelId={panelId}
          onSelect={setPicked}
          busy={(f) => f.models.some((m) => run.current === m.localName)}
          inUse={(f) => f.models.some(inUse)}
        />
        {!status.loading || installedBytes > 0 ? (
          <span className="text-sm tabular-nums text-content-muted">
            {installedBytes > 0 ? t`${formatBytes(installedBytes)} used` : t`Nothing installed`}
          </span>
        ) : null}
      </div>
      <div
        id={panelId}
        role="tabpanel"
        aria-label={familyLabel[family]}
        className="flex flex-col gap-2"
      >
        <p className="text-sm text-content-subtle">{familyNote[family]}</p>
        <div className="model-list">{shown.models.map(row)}</div>
      </div>
      <DeleteModelDialog
        dialogRef={dialogRef}
        model={deleting}
        onCancel={closeDelete}
        onConfirm={confirmDelete}
      />
    </>
  );
}

function useFamilyLabels(): Record<FamilyId, string> {
  const { t } = useLingui();
  return { whisper: t`Whisper`, parakeet: t`Parakeet` };
}

type Family = (typeof FAMILIES)[number];

interface FamilyTabsProps {
  selected: FamilyId;
  panelId: string;
  onSelect: (id: FamilyId) => void;
  /** A download of one of its models is in flight. */
  busy: (f: Family) => boolean;
  inUse: (f: Family) => boolean;
}

/** Same look as the engine tabs on the Transcription page. */
function FamilyTabs({ selected, panelId, onSelect, busy, inUse }: FamilyTabsProps) {
  const { t } = useLingui();
  const labels = useFamilyLabels();
  return (
    <div
      role="tablist"
      aria-label={t`Model family`}
      className="inline-flex w-fit flex-wrap gap-1 rounded-field border border-surface-border bg-surface-inset p-1"
    >
      {FAMILIES.map((f) => {
        const on = f.id === selected;
        return (
          <button
            key={f.id}
            type="button"
            role="tab"
            aria-selected={on}
            aria-controls={panelId}
            onClick={() => onSelect(f.id)}
            className={`inline-flex items-center gap-2 rounded-[calc(var(--radius-field)-3px)] px-4 py-2 text-sm transition-colors duration-fast ${
              on
                ? 'bg-surface-elevated font-medium text-content-strong shadow-sm ring-1 ring-surface-border-strong'
                : 'text-content hover:text-content-strong'
            }`}
          >
            {labels[f.id]}
            {busy(f) ? (
              <Loader2 size={12} className="animate-spin text-primary" aria-hidden />
            ) : (
              inUse(f) && <span className="size-1.5 rounded-full bg-primary" aria-hidden />
            )}
          </button>
        );
      })}
    </div>
  );
}

interface DeleteModelDialogProps {
  dialogRef: RefObject<HTMLDialogElement | null>;
  model: DeleteTarget | null;
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteModelDialog({ dialogRef, model, onCancel, onConfirm }: DeleteModelDialogProps) {
  const { t } = useLingui();
  return (
    <dialog ref={dialogRef} className="modal" onClose={onCancel}>
      <div className="modal-box">
        {model && (
          <>
            <h3 className="text-lg font-bold">{t`Delete ${model.name}?`}</h3>
            {model.inUse && (
              <p className="flex items-start gap-2 py-2 text-sm text-warning">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                {t`This model is in use by your current configuration. Dictation may not work until you reinstall it or pick another.`}
              </p>
            )}
            <div className="modal-action">
              <button type="button" className="btn btn-soft" onClick={onCancel}>
                {t`Cancel`}
              </button>
              <button type="button" className="btn btn-error" onClick={onConfirm}>
                {t`Delete model`}
              </button>
            </div>
          </>
        )}
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">{t`Close`}</button>
      </form>
    </dialog>
  );
}
