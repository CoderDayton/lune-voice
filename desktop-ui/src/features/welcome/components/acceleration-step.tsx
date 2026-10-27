// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Zap } from 'lucide-react';
import { formatBytes } from '../../../lib/format-bytes';
import type { GpuAccelerationOffer } from '../setup-logic';

/**
 * What the GPU offer comes to, in words. Promising speed the user does not
 * have yet would be a lie the first dictation exposes, so it says which of the
 * three situations this is: already accelerated, one restart away, or a
 * download setup is about to make.
 */
export function useGpuOfferDetail(offer: GpuAccelerationOffer): string {
  const { t } = useLingui();
  return offer === 'active'
    ? t`Your GPU is ready — use it for faster transcription.`
    : offer === 'needs-restart'
      ? t`Installed. Restart Lune Voice to start using your GPU.`
      : t`Your GPU qualifies. Setup downloads GPU support and turns it on.`;
}

interface AccelerationStepProps {
  offer: GpuAccelerationOffer;
  useGpu: boolean;
  downloadBytes: number;
  onToggle: (next: boolean) => void;
}

/**
 * The GPU screen. The header above it already says what accepting does; this
 * is the one decision, with the size of what it costs beside it.
 */
export function AccelerationStep({
  offer,
  useGpu,
  downloadBytes,
  onToggle,
}: AccelerationStepProps) {
  const { t } = useLingui();

  return (
    <label className="setup-card mx-auto flex w-full max-w-md cursor-pointer items-center gap-3 p-4">
      <span className="setup-tile bg-primary/15 text-primary">
        <Zap size={18} strokeWidth={1.8} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-content-strong">{t`Use GPU acceleration`}</span>
          {/* The download size is the one number worth weighing here. */}
          {offer === 'after-download' && downloadBytes > 0 && (
            <span className="badge badge-xs badge-soft tabular-nums text-content-subtle">
              +{formatBytes(downloadBytes)}
            </span>
          )}
        </span>
      </span>
      <input
        type="checkbox"
        className="toggle toggle-primary shrink-0"
        checked={useGpu}
        onChange={(e) => onToggle(e.target.checked)}
        aria-label={t`Use GPU acceleration`}
      />
    </label>
  );
}
