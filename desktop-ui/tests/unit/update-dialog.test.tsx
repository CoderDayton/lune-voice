// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, render } from '@testing-library/react';
import type { UpdateFlow } from '../../src/features/updates/use-update-flow';
import type { UpdateStateDto } from '../../src/lib/backend/generated/UpdateStateDto';

i18n.load('en', {});
i18n.activate('en');

let flow: UpdateFlow;
mock.module('../../src/features/updates/use-update-flow', () => ({
  useUpdateFlow: () => flow,
}));

// Imported after the mock above, so the dialog reads the mocked flow.
const { UpdateDialog } = await import('../../src/features/updates/update-dialog');

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

function withState(state: UpdateStateDto) {
  flow = {
    state,
    open: true,
    install: () => {},
    dismiss: () => {},
    retry: () => {},
  } as UpdateFlow;
}

describe('UpdateDialog', () => {
  test('shows a failed check with its error text', () => {
    withState({
      phase: 'check-failed',
      currentVersion: '0.1.0',
      availableVersion: null,
      notes: null,
      message: 'Could not check for updates.\n\nnetwork unreachable',
      downloadedBytes: 0,
      totalBytes: 0,
    });

    const { getByText } = render(
      <I18nProvider i18n={i18n}>
        <UpdateDialog />
      </I18nProvider>,
    );

    expect(getByText('Update check failed')).toBeDefined();
    expect(getByText(/network unreachable/)).toBeDefined();
  });

  test('puts the headline in plain text and only the error detail in the tinted block', () => {
    withState({
      phase: 'check-failed',
      currentVersion: '0.1.0',
      availableVersion: null,
      notes: null,
      message: 'Could not check for updates.\n\nCould not fetch a valid release JSON',
      downloadedBytes: 0,
      totalBytes: 0,
    });

    const { getByText, container } = render(
      <I18nProvider i18n={i18n}>
        <UpdateDialog />
      </I18nProvider>,
    );

    const block = container.querySelector('.alert') as HTMLElement;
    expect(block.classList.contains('alert-soft')).toBe(true);
    expect(block.textContent).toBe('Could not fetch a valid release JSON');
    expect(block.contains(getByText('Could not check for updates.'))).toBe(false);
  });

  test('keeps a one-paragraph failure whole inside the tinted block', () => {
    withState({
      phase: 'install-failed',
      currentVersion: '0.1.0',
      availableVersion: '0.2.0',
      notes: null,
      message: 'No package to replace.',
      downloadedBytes: 0,
      totalBytes: 0,
    });

    const { container } = render(
      <I18nProvider i18n={i18n}>
        <UpdateDialog />
      </I18nProvider>,
    );

    const block = container.querySelector('.alert') as HTMLElement;
    expect(block.classList.contains('alert-soft')).toBe(true);
    expect(block.textContent).toBe('No package to replace.');
  });
});
