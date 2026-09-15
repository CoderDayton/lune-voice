// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { DOT_COUNT, dotStyle } from './dot-style';

interface AudioDotsProps {
  level?: number; // 0.0–1.0 from real audio RMS
}

export function AudioDots({ level }: AudioDotsProps) {
  const { t } = useLingui();

  return (
    <div className="audio-dots" role="img" aria-label={t`Audio activity`}>
      {Array.from({ length: DOT_COUNT }, (_, i) => (
        <span key={`dot-${String(i)}`} className="audio-dot" style={dotStyle(level, i)} />
      ))}
    </div>
  );
}
