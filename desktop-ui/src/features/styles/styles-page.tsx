import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Check, Copy, Lock, Paintbrush, Pencil, Plus, Trash2 } from 'lucide-react';
import { useRef, useState } from 'react';
import {
  createCustomStyle,
  deleteCustomStyle,
  generateStyleExamples,
  saveWorkspaceSetting,
  updateCustomStyle,
  workspaceStore,
} from '../../store/workspace-store';
import type { StylePreset } from '../../store/workspace-types';

type ModalMode = 'create' | 'edit' | 'view';

interface ModalState {
  open: boolean;
  mode: ModalMode;
  preset: StylePreset | null;
  name: string;
  description: string;
  instructions: string;
  saving: boolean;
  generating: boolean;
}

const CLOSED_MODAL: ModalState = {
  open: false,
  mode: 'create',
  preset: null,
  name: '',
  description: '',
  instructions: '',
  saving: false,
  generating: false,
};

export function StylesPage() {
  const { t } = useLingui();
  const activePreset = useStore(workspaceStore, (s) => s.settings.stylePreset);
  const presets = useStore(workspaceStore, (s) => s.stylePresets);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);

  const [modal, setModal] = useState<ModalState>(CLOSED_MODAL);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previousPresetsRef = useRef(presets);

  function openModal(mode: ModalMode, preset?: StylePreset) {
    setModal({
      ...CLOSED_MODAL,
      open: true,
      mode,
      preset: preset ?? null,
      name: preset?.name ?? '',
      description: preset?.description ?? '',
      instructions: preset?.instructions ?? '',
    });
    requestAnimationFrame(() => dialogRef.current?.showModal());
  }

  function closeModal() {
    dialogRef.current?.close();
    setModal(CLOSED_MODAL);
  }

  function duplicateFromView() {
    if (!modal.preset) return;
    setModal((m) => ({ ...m, mode: 'create', preset: null, name: `${m.name} (copy)` }));
  }

  async function handleSave() {
    if (modal.saving || modal.generating) return;
    setModal((m) => ({ ...m, saving: true }));
    previousPresetsRef.current = presets;

    let ok: boolean;
    if (modal.mode === 'edit' && modal.preset) {
      ok = await updateCustomStyle(
        modal.preset.id,
        modal.name,
        modal.description,
        modal.instructions,
      );
    } else {
      ok = await createCustomStyle(modal.name, modal.description, modal.instructions);
    }

    if (!ok) {
      setModal((m) => ({ ...m, saving: false }));
      return;
    }

    let targetId: string | null = null;
    if (modal.mode === 'edit' && modal.preset) {
      targetId = modal.preset.id;
    } else {
      const cur = workspaceStore.state.stylePresets;
      targetId =
        cur.find(
          (p: StylePreset) =>
            !p.isBuiltin && !previousPresetsRef.current.some((ex) => ex.id === p.id),
        )?.id ?? null;
    }

    if (targetId) {
      setModal((m) => ({ ...m, saving: false, generating: true }));
      const genOk = await generateStyleExamples(targetId);
      setModal((m) => ({ ...m, generating: false }));
      if (!genOk) return;
    }

    closeModal();
  }

  async function handleDelete(id: string) {
    await deleteCustomStyle(id);
    setDeletingId(null);
  }

  const isViewOnly = modal.mode === 'view';
  const isEdit = modal.mode === 'edit';
  const instrLen = modal.instructions.length;
  const busy = modal.saving || modal.generating;

  return (
    <div className="page-stack">
      <div className="flex items-center gap-4 mb-2">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/8">
          <Paintbrush size={18} strokeWidth={1.8} className="text-primary/60" />
        </div>
        <div>
          <h2 className="text-xl font-semibold tracking-tight">{t`Writing styles`}</h2>
          <p className="text-sm text-content-subtle mt-0.5">
            {t`Choose the tone and formatting applied to your transcriptions.`}
          </p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {loading && presets.length === 0
          ? [0, 1, 2].map((i) => (
              <div
                key={i}
                className="card border rounded-box p-5 border-surface-border bg-surface-card"
              >
                <div className="skeleton h-5 w-24 mb-2" />
                <div className="skeleton h-3 w-full" />
              </div>
            ))
          : presets.map((preset) => {
              const isActive = activePreset === preset.id;
              const isConfirmingDelete = deletingId === preset.id;
              return (
                <div
                  key={preset.id}
                  className={`group card border rounded-box p-5 transition-colors relative ${
                    isActive
                      ? 'border-primary bg-primary/5'
                      : 'border-surface-border bg-surface-card hover:border-primary/40'
                  }`}
                >
                  <button
                    type="button"
                    className="absolute inset-0 rounded-box cursor-pointer focus-visible:outline-2 focus-visible:outline-primary"
                    aria-label={t`Select ${preset.name} style`}
                    onClick={() =>
                      !isConfirmingDelete && void saveWorkspaceSetting('stylePreset', preset.id)
                    }
                  />
                  <div className="relative pointer-events-none">
                    <div className="flex items-start justify-between mb-2 gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-medium truncate">{preset.name}</span>
                        {preset.isBuiltin && (
                          <span className="badge badge-ghost badge-sm shrink-0 gap-1">
                            <Lock size={10} />
                            {t`Built-in`}
                          </span>
                        )}
                      </div>
                      {isActive && !isConfirmingDelete && (
                        <Check size={16} className="text-primary shrink-0 mt-0.5" />
                      )}
                    </div>
                    <p className="text-sm text-content-subtle">{preset.description}</p>
                  </div>
                  <div className="relative mt-3 flex items-center justify-between min-h-[1.5rem]">
                    {isConfirmingDelete ? (
                      <div className="flex items-center gap-2 w-full pointer-events-auto">
                        <span className="text-xs text-content-muted">{t`Delete this style?`}</span>
                        <button
                          type="button"
                          className="btn btn-xs btn-error"
                          onClick={() => void handleDelete(preset.id)}
                        >{t`Delete`}</button>
                        <button
                          type="button"
                          className="btn btn-xs btn-ghost"
                          onClick={() => setDeletingId(null)}
                        >{t`Cancel`}</button>
                      </div>
                    ) : preset.isBuiltin ? (
                      <button
                        type="button"
                        className="pointer-events-auto opacity-0 group-hover:opacity-100 transition-opacity text-xs text-content-dim hover:text-content-subtle flex items-center gap-1"
                        onClick={() => openModal('view', preset)}
                      >
                        <Lock size={11} /> {t`View`}
                      </button>
                    ) : (
                      <div className="pointer-events-auto flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs gap-1"
                          onClick={() => openModal('edit', preset)}
                        >
                          <Pencil size={13} /> {t`Edit`}
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs text-error/70 hover:text-error hover:bg-error/8"
                          onClick={() => setDeletingId(preset.id)}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

        <button
          type="button"
          className="card border border-dashed rounded-box p-5 text-left border-surface-border bg-surface-card hover:border-primary/50 hover:bg-primary/3 transition-colors cursor-pointer group"
          onClick={() => openModal('create')}
        >
          <div className="flex items-center gap-3 text-content-dim group-hover:text-content-muted transition-colors">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-md border border-dashed border-current">
              <Plus size={14} />
            </div>
            <span className="text-sm font-medium">{t`Create new style`}</span>
          </div>
        </button>
      </div>

      {/* Modal */}
      <dialog ref={dialogRef} className="modal backdrop-blur-sm" onClose={closeModal}>
        <div className="modal-box max-w-lg overflow-hidden border border-surface-border">
          <h3 className="font-semibold text-lg mb-4">
            {isViewOnly ? t`View style` : isEdit ? t`Edit style` : t`Create style`}
          </h3>
          <div className="flex flex-col gap-4">
            <label className="flex flex-col gap-1.5">
              <span className="text-sm font-medium text-content-default">{t`Name`}</span>
              <input
                type="text"
                className="input input-bordered w-full"
                maxLength={64}
                placeholder={t`e.g. Meeting notes`}
                value={modal.name}
                readOnly={isViewOnly}
                onChange={(e) => setModal((m) => ({ ...m, name: e.target.value }))}
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-sm font-medium text-content-default">{t`Description`}</span>
              <input
                type="text"
                className="input input-bordered w-full"
                maxLength={256}
                placeholder={t`One-line summary shown on the card`}
                value={modal.description}
                readOnly={isViewOnly}
                onChange={(e) => setModal((m) => ({ ...m, description: e.target.value }))}
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-content-default">{t`Instructions`}</span>
                <span
                  className={`text-xs tabular-nums ${instrLen > 1900 ? 'text-warning' : 'text-content-dim'}`}
                >
                  {instrLen}/2048
                </span>
              </div>
              <textarea
                className="textarea textarea-bordered w-full min-h-32 resize-y leading-relaxed"
                maxLength={2048}
                placeholder={t`Describe how text should be formatted. E.g., "Use bullet points. Keep sentences short. Avoid jargon."`}
                value={modal.instructions}
                readOnly={isViewOnly}
                onChange={(e) => setModal((m) => ({ ...m, instructions: e.target.value }))}
              />
            </label>
          </div>
          <div className="modal-action mt-6">
            {isViewOnly ? (
              <>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={closeModal}
                >{t`Close`}</button>
                <button type="button" className="btn btn-primary gap-2" onClick={duplicateFromView}>
                  <Copy size={15} /> {t`Duplicate`}
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={closeModal}
                  disabled={busy}
                >{t`Cancel`}</button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!modal.name.trim() || !modal.instructions.trim() || busy}
                  onClick={() => void handleSave()}
                >
                  {busy && <span className="loading loading-spinner loading-sm" />}
                  {modal.generating ? t`Generating examples…` : t`Save`}
                </button>
              </>
            )}
          </div>
        </div>
        {!busy && (
          <form method="dialog" className="modal-backdrop">
            <button type="submit">{t`close`}</button>
          </form>
        )}
      </dialog>
    </div>
  );
}
