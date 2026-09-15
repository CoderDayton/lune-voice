// SPDX-License-Identifier: GPL-3.0-or-later

import type { ReactNode } from 'react';

type StatusVariant = 'success' | 'warning' | 'error' | 'info';

const DOT_STYLES: Record<StatusVariant, string> = {
  success: 'status-success',
  warning: 'status-warning',
  error: 'status-error',
  info: 'status-info',
};

interface StatusBadgeProps {
  variant: StatusVariant;
  children: ReactNode;
}

/** Status badge with dot indicator. Uses surface + content design tokens. */
export function StatusBadge({ variant, children }: StatusBadgeProps) {
  return (
    <div className="inline-flex items-center gap-1.5 sm:gap-2 rounded-full border border-surface-border bg-surface-panel px-3 py-1.5 text-sm">
      <span className={`status ${DOT_STYLES[variant]}`} />
      <span className="text-content">{children}</span>
    </div>
  );
}

interface InfoBadgeProps {
  children: ReactNode;
}

/** Metadata badge using surface + content tokens. */
export function InfoBadge({ children }: InfoBadgeProps) {
  return (
    <div className="inline-flex items-center rounded-full border border-surface-border bg-surface-panel px-3 py-1.5 text-sm text-content-muted">
      {children}
    </div>
  );
}
