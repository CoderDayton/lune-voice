import { useLingui } from '@lingui/react/macro';
import { Check, Lock, Pencil, Trash2 } from 'lucide-react';
import type { StylePreset } from '../../../store/workspace-types';

interface StylePresetCardProps {
  preset: StylePreset;
  isActive: boolean;
  isConfirmingDelete: boolean;
  onSelect: () => void;
  onView: () => void;
  onEdit: () => void;
  onRequestDelete: () => void;
  onCancelDelete: () => void;
  onConfirmDelete: () => void;
}

export function StylePresetCard({
  preset,
  isActive,
  isConfirmingDelete,
  onSelect,
  onView,
  onEdit,
  onRequestDelete,
  onCancelDelete,
  onConfirmDelete,
}: StylePresetCardProps) {
  const { t } = useLingui();

  return (
    <div
      className={`group card border rounded-box p-4 sm:p-5 relative transition-[background,border-color] duration-fast ease-standard ${
        isActive
          ? 'border-primary bg-primary/5'
          : 'border-surface-border bg-surface-card hover:border-primary/40 hover:bg-surface-card-hover'
      }`}
    >
      <button
        type="button"
        className="absolute inset-0 rounded-box cursor-pointer focus-visible:outline-2 focus-visible:outline-primary"
        aria-label={t`Select ${preset.name} style`}
        onClick={() => !isConfirmingDelete && onSelect()}
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
            <button type="button" className="btn btn-xs btn-error" onClick={onConfirmDelete}>
              {t`Delete`}
            </button>
            <button type="button" className="btn btn-xs btn-ghost" onClick={onCancelDelete}>
              {t`Cancel`}
            </button>
          </div>
        ) : preset.isBuiltin ? (
          <button
            type="button"
            className="pointer-events-auto opacity-0 group-hover:opacity-100 transition-opacity text-xs text-content-dim hover:text-content-subtle flex items-center gap-1"
            onClick={onView}
          >
            <Lock size={11} /> {t`View`}
          </button>
        ) : (
          <div className="pointer-events-auto flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button type="button" className="btn btn-ghost btn-xs gap-1" onClick={onEdit}>
              <Pencil size={13} /> {t`Edit`}
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-xs text-error/70 hover:text-error hover:bg-error/8"
              onClick={onRequestDelete}
            >
              <Trash2 size={13} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
