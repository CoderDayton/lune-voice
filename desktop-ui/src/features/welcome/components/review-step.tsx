// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check, Download, Zap } from 'lucide-react';
import type { ReactNode } from 'react';
import { formatBytes } from '../../../lib/format-bytes';
import type { ModelStatus } from '../setup-logic';

interface ReviewStepProps {
  models: ModelStatus[];
  selected: ReadonlySet<string>;
  asrLocalName: string;
  /** GPU support is part of this run. */
  willInstallGpu: boolean;
  gpuDownloadBytes: number;
}

/**
 * The last screen before anything downloads: every item the run would fetch or
 * has already, with its size. Nothing is chosen here, so the choices cannot
 * change under a download that has started.
 */
export function ReviewStep({
  models,
  selected,
  asrLocalName,
  willInstallGpu,
  gpuDownloadBytes,
}: ReviewStepProps) {
  const { t } = useLingui();
  // The engine's own model leads; the rest keep their library order.
  const chosen = models
    .filter((m) => selected.has(m.localName))
    .sort((a, b) => Number(b.localName === asrLocalName) - Number(a.localName === asrLocalName));

  return (
    <div className="setup-group mx-auto w-full max-w-xl">
      {chosen.map((m) => (
        <ReviewRow
          key={m.localName}
          icon={m.ready ? Check : Download}
          iconClass={m.ready ? 'bg-success/15 text-success' : 'bg-primary/10 text-primary'}
          name={m.name}
          description={m.description}
          trailing={
            m.ready ? (
              <span className="badge badge-xs badge-soft badge-success">{t`Installed`}</span>
            ) : m.downloadBytes !== null && m.downloadBytes > 0 ? (
              formatBytes(m.downloadBytes)
            ) : null
          }
        />
      ))}
      {willInstallGpu && (
        <ReviewRow
          icon={Zap}
          iconClass="bg-primary/10 text-primary"
          name={t`GPU acceleration`}
          trailing={gpuDownloadBytes > 0 ? formatBytes(gpuDownloadBytes) : null}
        />
      )}
    </div>
  );
}

interface ReviewRowProps {
  icon: typeof Check;
  iconClass: string;
  name: string;
  description?: string;
  trailing: ReactNode;
}

function ReviewRow({ icon: Icon, iconClass, name, description, trailing }: ReviewRowProps) {
  return (
    <div className="setup-row-divider flex items-center gap-3 px-4 py-3">
      <span className={`setup-tile ${iconClass}`}>
        <Icon size={18} strokeWidth={1.8} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-content-strong">{name}</span>
        {description && (
          <span className="mt-0.5 block text-xs leading-snug text-content-subtle">
            {description}
          </span>
        )}
      </span>
      {trailing !== null && (
        <span className="shrink-0 text-xs tabular-nums text-content-subtle">{trailing}</span>
      )}
    </div>
  );
}
