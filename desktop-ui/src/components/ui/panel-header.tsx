// SPDX-License-Identifier: GPL-3.0-or-later

import type { ComponentType, ReactNode } from 'react';

interface PanelHeaderProps {
  /** Leading icon, rendered inside the primary-tinted tile. */
  icon?: ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;
  /** Optional eyebrow above the title (label, badges). */
  eyebrow?: ReactNode;
  title: string;
  description?: string;
  /** Optional trailing controls aligned to the top-right. */
  actions?: ReactNode;
}

/**
 * The hero-style panel header: a primary-tinted icon tile beside a display
 * title and muted description. Shared by the styles hero card and every
 * settings section so headers read identically across the app.
 */
export function PanelHeader({
  icon: Icon,
  eyebrow,
  title,
  description,
  actions,
}: PanelHeaderProps) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex min-w-0 items-start gap-4">
        {Icon && (
          <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary ring-1 ring-primary/30">
            <Icon size={19} strokeWidth={1.75} />
          </div>
        )}
        <div className="min-w-0">
          {eyebrow && <div className="flex items-center gap-2">{eyebrow}</div>}
          <h2 className="mt-1 truncate font-display text-2xl font-semibold tracking-display text-content-strong">
            {title}
          </h2>
          {description && (
            <p className="mt-1 max-w-prose text-sm text-content-muted">{description}</p>
          )}
        </div>
      </div>
      {actions && <div className="shrink-0">{actions}</div>}
    </div>
  );
}
