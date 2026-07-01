import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Plus } from 'lucide-react';
import { useRef, useState } from 'react';
import type { FormatTogglesDto } from '../../lib/backend/types';
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
import { StyleHero } from './components/style-hero';
import { StylePresetCard } from './components/style-preset-card';

type ModalMode = 'create' | 'edit' | 'view';

interface ModalState {
  open: boolean;
  mode: ModalMode;
  preset: StylePreset | null;
  name: string;
  description: string;
  instructions: string;
  formatting: FormatTogglesDto;
  saving: boolean;
  generating: boolean;
}

const DEFAULT_FORMATTING: FormatTogglesDto = {
  keybinds: true,
  titleAbbreviations: true,
  autoCapitalization: true,
  brandCasing: true,
};

const CLOSED_MODAL: ModalState = {
  open: false,
  mode: 'create',
  preset: null,
  name: '',
  description: '',
  instructions: '',
  formatting: DEFAULT_FORMATTING,
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
      formatting: preset?.formatting ?? DEFAULT_FORMATTING,
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
        modal.formatting,
      );
    } else {
      ok = await createCustomStyle(
        modal.name,
        modal.description,
        modal.instructions,
        modal.formatting,
      );
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

  const activeStyle = presets.find((p) => p.id === activePreset) ?? null;
  const others = presets.filter((p) => p.id !== activeStyle?.id);
  const stylingEnabled = activePreset !== null;
  // Style applied when the user flips styling on. Prefer a builtin so it always resolves.
  const defaultStyleId = presets.find((p) => p.isBuiltin)?.id ?? presets[0]?.id ?? 'casual';

  return (
    <div className="page-stack">
      {loading && presets.length === 0 ? (
        <>
          <div className="skeleton h-32 w-full rounded-box" />
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="skeleton h-[9.75rem] w-full rounded-box" />
            ))}
          </div>
        </>
      ) : (
        <>
          <label className="flex cursor-pointer items-center justify-between rounded-box border border-surface-border bg-surface-inset/50 p-4">
            <div>
              <div className="text-sm font-medium text-content-strong">{t`Enable styling`}</div>
              <div className="mt-0.5 text-xs text-content-subtle">
                {t`Rewrite each transcript in a chosen style. When off, the raw transcript is delivered as-is.`}
              </div>
            </div>
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={stylingEnabled}
              onChange={(e) =>
                void saveWorkspaceSetting('stylePreset', e.target.checked ? defaultStyleId : null)
              }
            />
          </label>

          {stylingEnabled && activeStyle && (
            <StyleHero
              preset={activeStyle}
              onEdit={() => openModal('edit', activeStyle)}
              onView={() => openModal('view', activeStyle)}
            />
          )}

          <div
            className={`flex items-center justify-between pt-1 transition-opacity ${stylingEnabled ? '' : 'opacity-40'}`}
          >
            <span className="section-label">{t`Switch style`}</span>
            <span className="text-xs text-content-subtle">{t`${presets.length} presets`}</span>
          </div>

          <div
            className={`grid gap-4 transition-opacity sm:grid-cols-2 xl:grid-cols-3 ${stylingEnabled ? '' : 'pointer-events-none opacity-40'}`}
            aria-disabled={!stylingEnabled}
          >
            {others.map((preset) => (
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
              className="group flex min-h-[9.75rem] flex-col items-center justify-center gap-3 rounded-box border border-dashed border-surface-border bg-surface-inset/50 p-4 transition-[transform,border-color,background-color,box-shadow] duration-fast ease-standard hover:-translate-y-0.5 hover:border-primary/50 hover:bg-surface-card-hover hover:shadow-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary sm:p-5"
              onClick={() => openModal('create')}
            >
              <div className="flex size-10 items-center justify-center rounded-xl border border-dashed border-current text-content-subtle transition-colors group-hover:border-primary group-hover:text-primary">
                <Plus size={18} />
              </div>
              <div className="text-center">
                <div className="text-sm font-medium text-content-muted transition-colors group-hover:text-content-default">
                  {t`Create new style`}
                </div>
                <div className="mt-0.5 text-xs text-content-subtle">{t`Your own formatting preset`}</div>
              </div>
            </button>
          </div>
        </>
      )}
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
        formatting={modal.formatting}
        onFormattingChange={(key, value) =>
          setModal((m) => ({ ...m, formatting: { ...m.formatting, [key]: value } }))
        }
        onClose={closeModal}
        onSave={() => void handleSave()}
        onDuplicate={duplicateFromView}
      />
    </div>
  );
}
