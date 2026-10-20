// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Lock, Pencil, Trash2 } from 'lucide-react';
import type { StylePreset } from '../../../store/workspace-types';
import { presetDescription, presetName } from '../builtin-styles';
import { styleIcon } from '../style-icons';

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
  const { t, i18n } = useLingui();
  const name = presetName(i18n, preset);
  const Icon = styleIcon(preset);

  return (
    <div
      className={`group relative flex flex-col rounded-box border p-4 transition-[transform,box-shadow,border-color,background-color] duration-fast ease-standard sm:p-5 ${
        isActive
          ? 'border-primary/55 bg-linear-to-br from-primary/12 via-surface-card to-surface-card shadow-glow'
          : 'border-surface-border bg-surface-card shadow-card hover:-translate-y-0.5 hover:border-primary/40 hover:bg-surface-hover hover:shadow-elevated'
      }`}
    >
      <button
        type="button"
        className="absolute inset-0 z-0 cursor-pointer rounded-box focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        aria-label={t`Use the ${name} style`}
        onClick={() => !isConfirmingDelete && onSelect()}
      />

      <div className="pointer-events-none relative z-10 flex items-start gap-3">
        <div
          className={`flex size-9 shrink-0 items-center justify-center rounded-field transition-colors ${
            isActive
              ? 'bg-primary/20 text-primary'
              : 'bg-surface-inset text-content-muted group-hover:text-content-default'
          }`}
        >
          <Icon size={17} strokeWidth={1.75} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="truncate font-semibold text-content-strong">{name}</span>
            {preset.isBuiltin && (
              <span className="badge badge-soft badge-xs shrink-0 gap-1">
                <Lock size={9} />
                {t`Built-in`}
              </span>
            )}
          </div>
          <p className="mt-1 line-clamp-2 text-sm text-content-muted">
            {presetDescription(i18n, preset)}
          </p>
        </div>
        {isActive && (
          <span className="relative z-10 inline-flex shrink-0 items-center gap-1 rounded-full bg-primary/15 px-2 py-0.5 text-[0.6875rem] font-medium text-primary">
            <Check size={11} strokeWidth={2.5} />
            {t`Active`}
          </span>
        )}
      </div>

      {/* Actions only. A fixed-height card with a status footer left every
          card mostly empty; now the text sets the height. */}
      <div className="pointer-events-none relative z-10 mt-3 flex min-h-6 items-center justify-end">
        {isConfirmingDelete ? (
          <div className="pointer-events-auto flex items-center gap-1.5">
            <span className="text-xs text-content-muted">{t`Delete?`}</span>
            <button type="button" className="btn btn-error btn-xs" onClick={onConfirmDelete}>
              {t`Yes`}
            </button>
            <button type="button" className="btn btn-soft btn-xs" onClick={onCancelDelete}>
              {t`No`}
            </button>
          </div>
        ) : preset.isBuiltin ? (
          <button
            type="button"
            className="pointer-events-auto flex items-center gap-1 text-xs text-content-subtle opacity-0 transition-opacity hover:text-content-subtle group-hover:opacity-100"
            onClick={onView}
          >
            <Lock size={11} /> {t`View`}
          </button>
        ) : (
          <div className="pointer-events-auto flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button type="button" className="btn btn-soft btn-xs gap-1" onClick={onEdit}>
              <Pencil size={12} /> {t`Edit`}
            </button>
            <button
              type="button"
              className="btn btn-soft btn-error btn-xs"
              onClick={onRequestDelete}
              aria-label={t`Delete ${preset.name}`}
            >
              <Trash2 size={12} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
