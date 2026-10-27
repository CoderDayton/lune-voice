// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { Check } from 'lucide-react';
import type { WizardStep } from '../setup-logic';

interface WizardStepperProps {
  steps: WizardStep[];
  current: WizardStep;
}

/**
 * Where the user is in the wizard: a numbered dot per screen, finished ones
 * checked. Only the screens that exist for these choices are listed, so the
 * count never promises a step that will be skipped.
 */
export function WizardStepper({ steps, current }: WizardStepperProps) {
  const { t } = useLingui();
  const labels: Record<WizardStep, string> = {
    engine: t`Engine`,
    acceleration: t`GPU acceleration`,
    extras: t`Models`,
    review: t`Review`,
  };
  const at = steps.indexOf(current);

  return (
    <ol className="flex items-center justify-center gap-2 text-xs select-none">
      {steps.map((step, i) => {
        const done = i < at;
        const active = i === at;
        return (
          <li
            key={step}
            aria-current={active ? 'step' : undefined}
            className="flex items-center gap-2"
          >
            {i > 0 && (
              <span
                className={`h-px w-8 sm:w-12 ${done || active ? 'bg-primary/60' : 'bg-surface-border'}`}
                aria-hidden="true"
              />
            )}
            <span className="flex items-center gap-1.5">
              <span
                className={`flex size-5 items-center justify-center rounded-full text-[11px] font-medium tabular-nums transition-colors ${
                  done
                    ? 'bg-primary text-primary-content'
                    : active
                      ? 'border-2 border-primary text-primary'
                      : 'border border-surface-border-strong text-content-subtle'
                }`}
              >
                {done ? <Check size={11} strokeWidth={3.5} aria-hidden="true" /> : i + 1}
              </span>
              <span
                className={`hidden sm:inline ${active ? 'font-medium text-content-strong' : 'text-content-subtle'}`}
              >
                {labels[step]}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
