// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import {
  localizeText,
  type ModelChoice,
  type parakeetModels,
  type whisperModels,
} from '../data/settings-data';

type LocalModel = (typeof whisperModels)[number] | (typeof parakeetModels)[number];

export interface ModelOption {
  model: LocalModel;
  choice: ModelChoice;
  /** The model this row selects is the one in use. */
  active: boolean;
}

/**
 * The models of one engine that are on disk, as a single choice. Models that
 * are not installed are not offered: getting them is the Models page's job.
 */
export function InstalledModelPicker({
  options,
  onSelect,
}: {
  options: ModelOption[];
  onSelect: (choice: ModelChoice) => void;
}) {
  const { t, i18n } = useLingui();
  return (
    <fieldset className="flex flex-col divide-y divide-surface-border overflow-hidden rounded-box border border-surface-border bg-surface-card">
      <legend className="sr-only">{t`Installed models`}</legend>
      {options.map(({ model, choice, active }) => (
        <label
          key={model.value}
          className={`flex cursor-pointer items-center gap-4 border-s-2 px-5 py-4 transition-colors duration-fast ${
            active
              ? 'border-s-primary bg-primary/12'
              : 'border-s-transparent hover:bg-surface-hover'
          }`}
        >
          <input
            type="radio"
            name={`model-${choice.engine}`}
            className="radio radio-primary radio-sm"
            checked={active}
            onChange={() => onSelect(choice)}
          />
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="flex flex-wrap items-baseline gap-x-2 text-base font-medium text-content-strong">
              {localizeText(i18n, model.label)}
              <span className="text-sm font-normal tabular-nums text-content-muted">
                {model.size}
              </span>
              {active && <span className="text-xs font-medium text-primary">{t`In use`}</span>}
            </span>
            <span className="text-sm text-content-muted">
              {model.description === undefined ? '' : localizeText(i18n, model.description)}
            </span>
          </span>
        </label>
      ))}
    </fieldset>
  );
}
