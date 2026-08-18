import { useLingui } from '@lingui/react/macro';
import { Copy, RefreshCw } from 'lucide-react';
import type { RefObject } from 'react';
import { byteLength, LIMITS } from '../../../lib/backend/generated/limits';
import type { FormatTogglesDto } from '../../../lib/backend/types';

/** Fraction of the instructions budget at which the counter turns amber. */
const INSTRUCTIONS_WARN_FRACTION = 0.93;

interface StyleEditorModalProps {
  dialogRef: RefObject<HTMLDialogElement | null>;
  mode: 'create' | 'edit' | 'view';
  name: string;
  description: string;
  instructions: string;
  saving: boolean;
  generating: boolean;
  refreshing: boolean;
  fewShotDirty: boolean;
  llmReady: boolean;
  llmReason: string | null;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onInstructionsChange: (v: string) => void;
  formatting: FormatTogglesDto;
  onFormattingChange: (key: keyof FormatTogglesDto, value: boolean) => void;
  onClose: () => void;
  onSave: () => void;
  onRefreshExamples: () => void;
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
  refreshing,
  fewShotDirty,
  llmReady,
  llmReason,
  onNameChange,
  onDescriptionChange,
  onInstructionsChange,
  formatting,
  onFormattingChange,
  onClose,
  onSave,
  onRefreshExamples,
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
  // Measured in UTF-8 bytes, because that is the unit the backend rejects on.
  // A DOM `maxLength` counts UTF-16 code units instead, so it used to let a
  // 64-emoji name through the form and straight into a server-side rejection.
  const nameBytes = byteLength(name);
  const descBytes = byteLength(description);
  const instrBytes = byteLength(instructions);
  const tooLong =
    nameBytes > LIMITS.styleNameMaxBytes ||
    descBytes > LIMITS.styleDescMaxBytes ||
    instrBytes > LIMITS.styleInstructionsMaxBytes;
  const busy = saving || generating || refreshing;
  const canSave = !!name.trim() && !!instructions.trim() && !busy && !tooLong;

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
              placeholder={t`e.g. Meeting notes`}
              value={name}
              readOnly={isViewOnly}
              onChange={(e) => onNameChange(e.target.value)}
            />
            {nameBytes > LIMITS.styleNameMaxBytes ? (
              <span className="text-xs text-error">{t`Name is too long.`}</span>
            ) : null}
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-content-default">{t`Description`}</span>
            <input
              type="text"
              className="input w-full"
              placeholder={t`One-line summary shown on the card`}
              value={description}
              readOnly={isViewOnly}
              onChange={(e) => onDescriptionChange(e.target.value)}
            />
            {descBytes > LIMITS.styleDescMaxBytes ? (
              <span className="text-xs text-error">{t`Description is too long.`}</span>
            ) : null}
          </label>
          <label className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-content-default">{t`Instructions`}</span>
              <span
                className={`text-xs tabular-nums ${
                  instrBytes > LIMITS.styleInstructionsMaxBytes
                    ? 'text-error'
                    : instrBytes > LIMITS.styleInstructionsMaxBytes * INSTRUCTIONS_WARN_FRACTION
                      ? 'text-warning'
                      : 'text-content-subtle'
                }`}
              >
                {instrBytes}/{LIMITS.styleInstructionsMaxBytes}
              </span>
            </div>
            <textarea
              className={`textarea w-full min-h-32 resize-y leading-relaxed ${
                isViewOnly ? 'cursor-default bg-surface-inset/40 text-content-muted opacity-70' : ''
              }`}
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
          {!isViewOnly && (isEdit || !llmReady) && (
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium text-content-default">{t`Few-shot examples`}</span>
              {isEdit && (
                <p className="text-xs text-content-subtle">
                  {t`Examples teach the styling engine your style. Regenerate them after editing the prompt.`}
                </p>
              )}
              {isEdit && (
                <button
                  type="button"
                  className={`btn btn-sm gap-2 self-start ${
                    fewShotDirty && llmReady ? 'btn-primary' : 'btn-soft'
                  }`}
                  onClick={onRefreshExamples}
                  disabled={busy || !llmReady}
                  aria-busy={refreshing}
                >
                  {refreshing ? (
                    <span className="loading loading-spinner loading-sm" />
                  ) : (
                    <RefreshCw size={14} />
                  )}
                  {refreshing ? t`Generating examples…` : t`Refresh few-shot examples`}
                </button>
              )}
              {isEdit && fewShotDirty && llmReady && !refreshing && (
                <span className="text-xs text-warning">
                  {t`Examples are out of date since you edited the prompt.`}
                </span>
              )}
              {!llmReady && llmReason && <span className="text-xs text-warning">{llmReason}</span>}
            </div>
          )}
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
                {(saving || generating) && <span className="loading loading-spinner loading-sm" />}
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
