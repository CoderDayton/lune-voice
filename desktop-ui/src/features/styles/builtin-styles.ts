// SPDX-License-Identifier: GPL-3.0-or-later

import type { I18n, MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';

interface PresetText {
  id: string;
  name: string;
  description: string;
  isBuiltin: boolean;
}

const BUILTIN: Record<string, { name: MessageDescriptor; description: MessageDescriptor }> = {
  casual: { name: msg`Casual`, description: msg`Texting, messaging, chat` },
  professional: { name: msg`Professional`, description: msg`Clear workplace communication` },
  formal: { name: msg`Formal`, description: msg`Official correspondence and documents` },
  technical: { name: msg`Technical`, description: msg`Code and developer communication` },
  writing: { name: msg`Writing`, description: msg`Long-form content — articles, docs, essays` },
};

function builtin(preset: PresetText) {
  return preset.isBuiltin ? BUILTIN[preset.id] : undefined;
}

export function presetName(i18n: I18n, preset: PresetText): string {
  const known = builtin(preset);
  return known ? i18n._(known.name) : preset.name;
}

export function presetDescription(i18n: I18n, preset: PresetText): string {
  const known = builtin(preset);
  return known ? i18n._(known.description) : preset.description;
}

export function styleLabel(i18n: I18n, id: string, presets: readonly PresetText[]): string {
  const known = BUILTIN[id];
  if (known) return i18n._(known.name);
  return presets.find((preset) => preset.id === id)?.name ?? id;
}
