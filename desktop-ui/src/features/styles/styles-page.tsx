import { useLingui } from '@lingui/react/macro';
import { useStore } from '@tanstack/react-store';
import { Plus } from 'lucide-react';
import { useRef, useState } from 'react';
import { getLlmReadiness } from '../../lib/backend/llm';
import type { FormatTogglesDto } from '../../lib/backend/types';
import {
  createCustomStyle,
  deleteCustomStyle,
  generateStyleExamples,
  saveWorkspaceSettings,
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
  refreshing: boolean;
  fewShotDirty: boolean;
  llmReady: boolean;
  llmReason: string | null;
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
  refreshing: false,
  fewShotDirty: false,
  llmReady: true,
  llmReason: null,
};

export function StylesPage() {
  const { t } = useLingui();
  const activePreset = useStore(workspaceStore, (s) => s.settings.stylePreset);
  const presets = useStore(workspaceStore, (s) => s.stylePresets);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);
  // Read the current styling-provider fields for readiness checks. Read from
  // the store at call time (not via a subscription) so checks always see the
  // latest saved values.
  const readLlmSettings = () => {
    const s = workspaceStore.state.settings;
    return {
      llmEnabled: s.llmEnabled,
      openaiLlmApiKey: s.openaiLlmApiKey,
      openaiLlmModel: s.openaiLlmModel,
    };
  };

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
      fewShotDirty: preset?.fewShotDirty ?? false,
    });
    requestAnimationFrame(() => dialogRef.current?.showModal());
    // Resolve styling-engine readiness in the background so the editor can gate
    // example generation (disable refresh + explain what to set up).
    if (mode !== 'view') {
      const r = getLlmReadiness(readLlmSettings());
      setModal((m) => ({ ...m, llmReady: r.ready, llmReason: r.reason }));
    }
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
    if (modal.saving || modal.generating || modal.refreshing) return;
    setModal((m) => ({ ...m, saving: true }));

    // Editing an existing style: save only. Changing the prompt marks its
    // examples stale (backend); the user refreshes them via the button.
    if (modal.mode === 'edit' && modal.preset) {
      const ok = await updateCustomStyle(
        modal.preset.id,
        modal.name,
        modal.description,
        modal.instructions,
        modal.formatting,
      );
      if (!ok) {
        setModal((m) => ({ ...m, saving: false }));
        return;
      }
      closeModal();
      return;
    }

    // Creating a new style.
    previousPresetsRef.current = presets;
    const ok = await createCustomStyle(
      modal.name,
      modal.description,
      modal.instructions,
      modal.formatting,
    );
    if (!ok) {
      setModal((m) => ({ ...m, saving: false }));
      return;
    }

    const created = workspaceStore.state.stylePresets.find(
      (p: StylePreset) => !p.isBuiltin && !previousPresetsRef.current.some((ex) => ex.id === p.id),
    );
    if (!created) {
      closeModal();
      return;
    }

    // The style now exists — flip the editor to edit mode over the persisted
    // preset. The Regenerate button becomes available exactly when the style
    // is created, and stays disabled (llmReady=false) until the engine
    // validates below.
    setModal((m) => ({
      ...m,
      saving: false,
      generating: true,
      mode: 'edit',
      preset: created,
      fewShotDirty: true,
      llmReady: false,
      llmReason: null,
    }));

    // Validate the provider is configured before generating and applying the
    // style.
    const readiness = getLlmReadiness(readLlmSettings());
    if (!readiness.ready) {
      // Style is created but stays dirty; Regenerate remains disabled with the
      // reason shown until the engine is set up and initialized.
      setModal((m) => ({ ...m, generating: false, llmReady: false, llmReason: readiness.reason }));
      return;
    }
    setModal((m) => ({ ...m, llmReady: true }));

    const genOk = await generateStyleExamples(created.id);
    const updated = workspaceStore.state.stylePresets.find((p: StylePreset) => p.id === created.id);
    setModal((m) => ({ ...m, generating: false, fewShotDirty: updated?.fewShotDirty ?? true }));
    // Keep the editor open if generation failed so the user notices and can
    // retry via the now-enabled Regenerate button (created + validated).
    if (!genOk) return;

    closeModal();
  }

  async function handleRefresh() {
    const preset = modal.preset;
    if (!preset || modal.refreshing) return;
    // Re-validate configuration at click time — settings may have changed
    // since the modal opened.
    setModal((m) => ({ ...m, refreshing: true }));
    const readiness = getLlmReadiness(readLlmSettings());
    if (!readiness.ready) {
      setModal((m) => ({ ...m, refreshing: false, llmReady: false, llmReason: readiness.reason }));
      return;
    }
    setModal((m) => ({ ...m, llmReady: true, llmReason: null }));
    await generateStyleExamples(preset.id);
    const updated = workspaceStore.state.stylePresets.find((p: StylePreset) => p.id === preset.id);
    setModal((m) => ({ ...m, refreshing: false, fewShotDirty: updated?.fewShotDirty ?? false }));
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
                void saveWorkspaceSettings({
                  stylePreset: e.target.checked ? defaultStyleId : null,
                  llmEnabled: e.target.checked,
                })
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
                onSelect={() =>
                  void saveWorkspaceSettings({ stylePreset: preset.id, llmEnabled: true })
                }
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
        refreshing={modal.refreshing}
        fewShotDirty={modal.fewShotDirty}
        llmReady={modal.llmReady}
        llmReason={modal.llmReason}
        onNameChange={(v) => setModal((m) => ({ ...m, name: v }))}
        onDescriptionChange={(v) => setModal((m) => ({ ...m, description: v }))}
        onInstructionsChange={(v) => setModal((m) => ({ ...m, instructions: v }))}
        formatting={modal.formatting}
        onFormattingChange={(key, value) =>
          setModal((m) => ({ ...m, formatting: { ...m.formatting, [key]: value } }))
        }
        onClose={closeModal}
        onSave={() => void handleSave()}
        onRefreshExamples={() => void handleRefresh()}
        onDuplicate={duplicateFromView}
      />
    </div>
  );
}
