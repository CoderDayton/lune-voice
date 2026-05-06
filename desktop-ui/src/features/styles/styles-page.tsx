import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Paintbrush, Plus } from 'lucide-react';
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
import { StyleEditorModal } from './components/style-editor-modal';
import { StylePresetCard } from './components/style-preset-card';

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
          : presets.map((preset) => (
              <StylePresetCard
                key={preset.id}
                preset={preset}
                isActive={activePreset === preset.id}
                isConfirmingDelete={deletingId === preset.id}
                onSelect={() => void saveWorkspaceSetting('stylePreset', preset.id)}
                onView={() => openModal('view', preset)}
                onEdit={() => openModal('edit', preset)}
                onRequestDelete={() => setDeletingId(preset.id)}
                onCancelDelete={() => setDeletingId(null)}
                onConfirmDelete={() => void handleDelete(preset.id)}
              />
            ))}

        <button
          type="button"
          className="card border border-dashed rounded-box p-5 text-left border-surface-border bg-surface-card hover:border-primary/50 hover:bg-surface-card-hover transition-[background,border-color] duration-fast ease-standard cursor-pointer group focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
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
      <StyleEditorModal
        dialogRef={dialogRef}
        mode={modal.mode}
        name={modal.name}
        description={modal.description}
        instructions={modal.instructions}
        saving={modal.saving}
        generating={modal.generating}
        onNameChange={(v) => setModal((m) => ({ ...m, name: v }))}
        onDescriptionChange={(v) => setModal((m) => ({ ...m, description: v }))}
        onInstructionsChange={(v) => setModal((m) => ({ ...m, instructions: v }))}
        onClose={closeModal}
        onSave={() => void handleSave()}
        onDuplicate={duplicateFromView}
      />
    </div>
  );
}
