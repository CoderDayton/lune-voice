// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { createRef } from 'react';
import { StyleEditorModal } from '../../src/features/styles/components/style-editor-modal';
import type { FormatTogglesDto } from '../../src/lib/backend/types';

i18n.load('en', {});
i18n.activate('en');

beforeAll(() => {
  GlobalRegistrator.register();
});

afterEach(() => {
  cleanup();
});

afterAll(async () => {
  cleanup();
  // Let React finish its last scheduled render before the DOM globals go.
  await new Promise((resolve) => setTimeout(resolve, 50));
  await GlobalRegistrator.unregister();
});

function renderModal(
  brandCasing: boolean | null,
  onBrandCasingChange: (value: boolean | null) => void = () => {},
  mode: 'edit' | 'view' = 'edit',
) {
  const formatting: FormatTogglesDto = {
    keybinds: true,
    titleAbbreviations: true,
    autoCapitalization: true,
    brandCasing,
  };
  const view = render(
    <I18nProvider i18n={i18n}>
      <StyleEditorModal
        dialogRef={createRef<HTMLDialogElement>()}
        mode={mode}
        name="Mine"
        description=""
        instructions="Keep it short."
        saving={false}
        generating={false}
        refreshing={false}
        fewShotDirty={false}
        llmReady
        llmReason={null}
        onNameChange={() => {}}
        onDescriptionChange={() => {}}
        onInstructionsChange={() => {}}
        formatting={formatting}
        onFormattingChange={() => {}}
        onBrandCasingChange={onBrandCasingChange}
        onClose={() => {}}
        onSave={() => {}}
        onRefreshExamples={() => {}}
        onDuplicate={() => {}}
      />
    </I18nProvider>,
  );
  return view.getByLabelText('Brand casing') as HTMLSelectElement;
}

describe('StyleEditorModal brand casing', () => {
  test('shows follow, on and off for the three stored values', () => {
    expect(renderModal(null).value).toBe('inherit');
    cleanup();
    expect(renderModal(true).value).toBe('on');
    cleanup();
    expect(renderModal(false).value).toBe('off');
  });

  test('reports the chosen value, with follow as null', () => {
    const seen: (boolean | null)[] = [];
    const select = renderModal(null, (value) => seen.push(value));

    fireEvent.change(select, { target: { value: 'off' } });
    fireEvent.change(select, { target: { value: 'on' } });
    fireEvent.change(select, { target: { value: 'inherit' } });

    expect(seen).toEqual([false, true, null]);
  });

  test('cannot be changed on a built-in style', () => {
    expect(renderModal(null, () => {}, 'view').disabled).toBe(true);
  });
});
