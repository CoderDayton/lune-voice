// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { AlertTriangle, Check, Clock, Download, Loader2, Trash2 } from 'lucide-react';
import { type ReactNode, type RefObject, useEffect, useId, useRef, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import { formatBytes } from '../../../lib/format-bytes';
import { DownloadChecklist } from '../../model-downloads/components/download-checklist';
import { runChecklist } from '../../model-downloads/download-run';
import { modelDownloads, useModelDownloads } from '../../model-downloads/model-downloads';
import { localizeText, parakeetModels, whisperModels } from '../data/settings-data';
import type { ModelStatusApi } from '../hooks/use-model-status';
import { SettingsBlock, SettingsRow } from './section';

type LocalModel = (typeof whisperModels)[number] | (typeof parakeetModels)[number];

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
 * Every on-device model in one place: tick the ones you want and download them
 * together, watch the shared download checklist while they come in, and delete
 * the ones you no longer need. Which installed model is used is picked on the
 * Transcription page.
 */
export function ModelLibrary({ status }: { status: ModelStatusApi }) {
  const { t, i18n } = useLingui();
  const run = useModelDownloads();
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  useResumeDownloads(status);

  const label = (m: LocalModel) => localizeText(i18n, m.label);
  const info = (m: LocalModel) => status.byName.get(m.localName);
  const isReady = (m: LocalModel) => info(m)?.ready ?? false;
  const bytesOf = (m: LocalModel) => info(m)?.downloadBytes ?? 0;
  const queued = (m: LocalModel) =>
    run.status === 'running' && run.models.some((r) => r.localName === m.localName);
  const selectable = (m: LocalModel) => !isReady(m) && !queued(m);

  const all: LocalModel[] = [...whisperModels, ...parakeetModels];
  const missing = all.filter(selectable);
  const wanted = missing.filter((m) => picked.has(m.localName));
  const wantedBytes = wanted.reduce((sum, m) => sum + bytesOf(m), 0);
  const allPicked = missing.length > 0 && wanted.length === missing.length;

  const toggle = (localName: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(localName)) next.delete(localName);
      else next.add(localName);
      return next;
    });

  function downloadPicked() {
    modelDownloads.start(
      wanted.map((m) => ({
        localName: m.localName,
        name: info(m)?.name ?? label(m),
        bytes: bytesOf(m),
      })),
    );
    setPicked(new Set());
  }

  function closeDelete() {
    dialogRef.current?.close();
    setDeleting(null);
  }

  function confirmDelete() {
    if (deleting) void status.remove(deleting.localName);
    closeDelete();
  }

  const stateOf = (m: LocalModel): RowState =>
    isReady(m)
      ? 'installed'
      : run.current === m.localName
        ? 'downloading'
        : queued(m)
          ? 'waiting'
          : 'available';

  const row = (m: LocalModel) => {
    const inUse = (info(m)?.inUse ?? false) && isReady(m);
    return (
      <ModelRow
        key={m.value}
        label={label(m)}
        description={m.description === undefined ? '' : localizeText(i18n, m.description)}
        size={bytesOf(m) > 0 ? formatBytes(bytesOf(m)) : m.size}
        state={stateOf(m)}
        inUse={inUse}
        checked={picked.has(m.localName)}
        onToggle={() => toggle(m.localName)}
        onDelete={() => {
          setDeleting({ localName: m.localName, name: label(m), inUse });
          dialogRef.current?.showModal();
        }}
      />
    );
  };

  const installedBytes = status.models.reduce(
    (sum, m) => sum + (m.ready && m.sizeBytes != null ? m.sizeBytes : 0),
    0,
  );

  return (
    <>
      {run.status !== 'idle' && <RunPanel run={run} />}
      <SettingsBlock label={t`Whisper`} description={t`Nearly 100 languages`}>
        <ModelList>{whisperModels.map(row)}</ModelList>
      </SettingsBlock>
      <SettingsBlock label={t`Parakeet`} description={t`25 European languages`}>
        <ModelList>{parakeetModels.map(row)}</ModelList>
      </SettingsBlock>
      {!status.loading && missing.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-box border border-surface-border bg-surface-card px-5 py-3">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() =>
              setPicked(allPicked ? new Set() : new Set(missing.map((m) => m.localName)))
            }
          >
            {allPicked ? t`Clear selection` : t`Select all`}
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm gap-1.5"
            disabled={wanted.length === 0}
            onClick={downloadPicked}
          >
            <Download size={14} />
            {wanted.length === 0
              ? t`Download`
              : wantedBytes > 0
                ? t`Download ${wanted.length} selected · ${formatBytes(wantedBytes)}`
                : t`Download ${wanted.length} selected`}
          </button>
        </div>
      )}
      {!status.loading && (
        <SettingsRow
          label={t`Disk usage`}
          control={
            <span className="text-sm font-medium tabular-nums text-content">
              {installedBytes > 0 ? formatBytes(installedBytes) : t`Nothing installed`}
            </span>
          }
        />
      )}
      <DeleteModelDialog
        dialogRef={dialogRef}
        model={deleting}
        onCancel={closeDelete}
        onConfirm={confirmDelete}
      />
    </>
  );
}

/** The shared download checklist, with the controls its state calls for. */
function RunPanel({ run }: { run: ReturnType<typeof useModelDownloads> }) {
  const { t } = useLingui();
  return (
    <div className="flex flex-col gap-3">
      <DownloadChecklist checklist={runChecklist(run, null)} />
      {run.status === 'failed' && (
        <div role="alert" className="alert alert-warning alert-soft gap-2 text-sm select-text">
          <AlertTriangle size={16} className="shrink-0" />
          <span>{run.error}</span>
        </div>
      )}
      <div className="flex justify-end gap-2">
        {run.status === 'running' && (
          <button
            type="button"
            className="btn btn-soft btn-sm"
            onClick={() => modelDownloads.cancel()}
          >
            {t`Cancel`}
          </button>
        )}
        {run.status === 'failed' && (
          <>
            <button
              type="button"
              className="btn btn-soft btn-sm"
              onClick={() => modelDownloads.dismiss()}
            >
              {t`Dismiss`}
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => modelDownloads.retry()}
            >
              {t`Retry`}
            </button>
          </>
        )}
        {run.status === 'settled' && (
          <button
            type="button"
            className="btn btn-soft btn-sm"
            onClick={() => modelDownloads.dismiss()}
          >
            {t`Done`}
          </button>
        )}
      </div>
    </div>
  );
}

function ModelList({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col divide-y divide-surface-border overflow-hidden rounded-box border border-surface-border bg-surface-card">
      {children}
    </div>
  );
}

type RowState = 'installed' | 'available' | 'waiting' | 'downloading';

interface ModelRowProps {
  label: string;
  description: string;
  size: string;
  state: RowState;
  inUse: boolean;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
}

function ModelRow({
  label,
  description,
  size,
  state,
  inUse,
  checked,
  onToggle,
  onDelete,
}: ModelRowProps) {
  const { t } = useLingui();
  const id = useId();
  const available = state === 'available';

  const text = (
    <>
      <span className="flex flex-wrap items-baseline gap-x-2 text-base font-medium text-content-strong">
        {label}
        <span className="text-sm font-normal tabular-nums text-content-muted">{size}</span>
        {inUse && <span className="text-xs font-medium text-primary">{t`In use`}</span>}
      </span>
      <span className="text-sm text-content-muted">{description}</span>
    </>
  );

  return (
    <div
      className={`flex items-center gap-4 border-s-2 px-5 py-4 transition-colors duration-fast ${
        inUse ? 'border-s-primary bg-primary/12' : 'border-s-transparent hover:bg-surface-hover'
      }`}
    >
      <span className="flex size-5 shrink-0 items-center justify-center">
        {available ? (
          <input
            id={id}
            type="checkbox"
            className="checkbox checkbox-primary checkbox-sm"
            checked={checked}
            onChange={onToggle}
          />
        ) : state === 'installed' ? (
          <Check size={16} strokeWidth={2.5} className="text-success" />
        ) : state === 'downloading' ? (
          <Loader2 size={16} className="animate-spin text-primary" />
        ) : (
          <Clock size={16} className="text-content-subtle" />
        )}
      </span>
      {available ? (
        <label htmlFor={id} className="flex min-w-0 flex-1 cursor-pointer flex-col gap-0.5">
          {text}
        </label>
      ) : (
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">{text}</div>
      )}
      <span className="flex min-w-24 shrink-0 items-center justify-end gap-1 text-sm font-medium">
        {state === 'installed' && (
          <>
            <span className="text-success">{t`Installed`}</span>
            <button
              type="button"
              className="btn btn-ghost btn-sm btn-square text-content-muted hover:text-error"
              onClick={onDelete}
              aria-label={t`Delete`}
              title={t`Delete`}
            >
              <Trash2 size={14} />
            </button>
          </>
        )}
        {state === 'downloading' && <span className="text-primary">{t`Downloading…`}</span>}
        {state === 'waiting' && <span className="text-content-muted">{t`Waiting`}</span>}
      </span>
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
