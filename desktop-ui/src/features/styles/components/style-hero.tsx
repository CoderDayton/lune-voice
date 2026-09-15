// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Lock, Pencil } from 'lucide-react';
import { PanelHeader } from '../../../components/ui/panel-header';
import type { StylePreset } from '../../../store/workspace-types';
import { styleIcon } from '../style-icons';

interface StyleHeroProps {
  preset: StylePreset;
  onEdit: () => void;
  onView: () => void;
}

export function StyleHero({ preset, onEdit, onView }: StyleHeroProps) {
  const { t } = useLingui();
  const Icon = styleIcon(preset);
  const fmt = preset.formatting;
  const chips = [
    { on: fmt.keybinds, label: t`Shortcuts` },
    { on: fmt.autoCapitalization, label: t`Capitals` },
    { on: fmt.titleAbbreviations, label: t`Titles` },
    { on: fmt.brandCasing, label: t`Brands` },
  ];

  return (
    <section className="relative overflow-hidden rounded-box border border-primary/30 bg-surface-card p-5 shadow-glow sm:p-6">
      <PanelHeader
        icon={Icon}
        eyebrow={
          <>
            <span className="section-label text-primary">{t`Active style`}</span>
            {preset.isBuiltin && (
              <span className="badge badge-soft badge-xs gap-1">
                <Lock size={9} /> {t`Built-in`}
              </span>
            )}
          </>
        }
        title={preset.name}
        description={preset.description}
        actions={
          preset.isBuiltin ? (
            <button type="button" className="btn btn-soft btn-sm gap-2" onClick={onView}>
              <Lock size={14} /> {t`View`}
            </button>
          ) : (
            <button type="button" className="btn btn-primary btn-sm gap-2" onClick={onEdit}>
              <Pencil size={14} /> {t`Edit`}
            </button>
          )
        }
      />

      <div className="relative mt-5 flex flex-wrap items-center gap-2">
        <span className="mr-1 text-xs text-content-subtle">{t`Formatting`}</span>
        {chips.map((c) => (
          <span
            key={c.label}
            className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs transition-colors ${
              c.on
                ? 'border-primary/30 bg-primary/10 text-primary'
                : 'border-surface-border bg-surface-inset text-content-subtle line-through'
            }`}
          >
            {c.label}
          </span>
        ))}
      </div>
    </section>
  );
}
