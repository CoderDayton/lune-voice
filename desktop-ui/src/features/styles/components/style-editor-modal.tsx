import { useLingui } from '@lingui/react/macro';
import { Copy } from 'lucide-react';
import type { RefObject } from 'react';
import type { FormatTogglesDto } from '../../../lib/backend/types';

interface StyleEditorModalProps {
  dialogRef: RefObject<HTMLDialogElement | null>;
  mode: 'create' | 'edit' | 'view';
  name: string;
  description: string;
  instructions: string;
  saving: boolean;
  generating: boolean;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onInstructionsChange: (v: string) => void;
  formatting: FormatTogglesDto;
  onFormattingChange: (key: keyof FormatTogglesDto, value: boolean) => void;
  onClose: () => void;
  onSave: () => void;
  onDuplicate: () => void;
}

export function StyleEditorModal({
  dialogRef,
  mode,
  name,
  description,
  instructions,
  saving,
  generating,
  onNameChange,
  onDescriptionChange,
  onInstructionsChange,
  formatting,
  onFormattingChange,
  onClose,
  onSave,
  onDuplicate,
}: StyleEditorModalProps) {
  const { t } = useLingui();
  const formatToggleRows: { key: keyof FormatTogglesDto; label: string }[] = [
    { key: 'keybinds', label: t`Keyboard shortcuts ("Super H" → "Super+H")` },
    { key: 'titleAbbreviations', label: t`Title abbreviations ("Mister" → "Mr.")` },
    { key: 'autoCapitalization', label: t`Auto-capitalize sentences` },
    { key: 'brandCasing', label: t`Brand casing ("github" → "GitHub")` },
  ];
  const isViewOnly = mode === 'view';
  const isEdit = mode === 'edit';
  const instrLen = instructions.length;
  const busy = saving || generating;
  const canSave = !!name.trim() && !!instructions.trim() && !busy;

  return (
    <dialog ref={dialogRef} className="modal" onClose={onClose}>
      <div className="modal-box flex max-h-[85vh] max-w-lg flex-col overflow-hidden border border-surface-border p-5 sm:p-6">
        <h3 className="mb-4 shrink-0 font-semibold text-lg">
          {isViewOnly ? t`View style` : isEdit ? t`Edit style` : t`Create style`}
        </h3>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Name`}</span>
            <input
              type="text"
              className="input w-full"
              maxLength={64}
              placeholder={t`e.g. Meeting notes`}
              value={name}
              readOnly={isViewOnly}
              onChange={(e) => onNameChange(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Description`}</span>
            <input
              type="text"
              className="input w-full"
              maxLength={256}
              placeholder={t`One-line summary shown on the card`}
              value={description}
              readOnly={isViewOnly}
              onChange={(e) => onDescriptionChange(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-content-default">{t`Instructions`}</span>
              <span
                className={`text-xs tabular-nums ${instrLen > 1900 ? 'text-warning' : 'text-content-subtle'}`}
              >
                {instrLen}/2048
              </span>
            </div>
            <textarea
              className="textarea w-full min-h-32 resize-y leading-relaxed"
              maxLength={2048}
              placeholder={t`Describe how text should be formatted. E.g., "Use bullet points. Keep sentences short. Avoid jargon."`}
              value={instructions}
              readOnly={isViewOnly}
              onChange={(e) => onInstructionsChange(e.target.value)}
            />
          </label>
          <div className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Formatting`}</span>
            <p className="text-xs text-content-subtle">
              {t`Deterministic fixes applied before any AI styling.`}
            </p>
            <div className="flex flex-col divide-y divide-surface-border rounded-box border border-surface-border">
              {formatToggleRows.map(({ key, label }) => (
                <label key={key} className="flex items-center justify-between gap-3 px-3 py-2">
                  <span className="text-sm text-content-default">{label}</span>
                  <input
                    type="checkbox"
                    className="toggle toggle-primary toggle-sm"
                    checked={formatting[key]}
                    disabled={isViewOnly}
                    onChange={(e) => onFormattingChange(key, e.target.checked)}
                    aria-label={label}
                  />
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="modal-action mt-6 shrink-0">
          {isViewOnly ? (
            <>
              <button type="button" className="btn btn-soft" onClick={onClose}>
                {t`Close`}
              </button>
              <button type="button" className="btn btn-primary gap-2" onClick={onDuplicate}>
                <Copy size={15} /> {t`Duplicate`}
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn btn-soft" onClick={onClose} disabled={busy}>
                {t`Cancel`}
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={!canSave}
                onClick={onSave}
              >
                {busy && <span className="loading loading-spinner loading-sm" />}
                {generating ? t`Generating examples…` : t`Save`}
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
  );
}
