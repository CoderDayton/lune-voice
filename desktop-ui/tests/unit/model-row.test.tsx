// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { ModelRow, type RowState } from '../../src/features/settings/components/model-row';

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
  await new Promise((resolve) => setTimeout(resolve, 50));
  await GlobalRegistrator.unregister();
});

const MB = 1024 * 1024;

function show(state: RowState, inUse = false) {
  const handlers = {
    onDownload: mock(() => {}),
    onCancel: mock(() => {}),
    onRetry: mock(() => {}),
    onDelete: mock(() => {}),
  };
  const view = render(
    <I18nProvider i18n={i18n}>
      <ModelRow
        label="Medium"
        description="Balanced accuracy and speed."
        size="539 MB"
        state={state}
        inUse={inUse}
        {...handlers}
      />
    </I18nProvider>,
  );
  return { view, ...handlers };
}

describe('ModelRow', () => {
  test('a missing model has its own Download button', () => {
    const { view, onDownload } = show({ kind: 'available' });
    fireEvent.click(view.getByRole('button', { name: 'Download Medium' }));
    expect(onDownload).toHaveBeenCalledTimes(1);
    expect(view.queryByRole('checkbox')).toBeNull();
  });

  test('a download in flight shows its progress in the row and can be cancelled', () => {
    const { view, onCancel } = show({ kind: 'downloading', done: 231 * MB, total: 539 * MB });
    const bar = view.getByRole('progressbar');
    expect(bar.getAttribute('value')).toBe(String(231 * MB));
    expect(bar.getAttribute('max')).toBe(String(539 * MB));
    expect(view.container.textContent).toContain('231 MB / 539 MB');
    expect(view.container.textContent).toContain('43%');
    fireEvent.click(view.getByRole('button', { name: 'Cancel download of Medium' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  test('a queued model says so and can be taken off the queue', () => {
    const { view, onCancel } = show({ kind: 'queued' });
    expect(view.container.textContent).toContain('Queued');
    fireEvent.click(view.getByRole('button', { name: 'Cancel download of Medium' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  test('a failed download shows why, in the row, with Retry', () => {
    const { view, onRetry, onCancel } = show({ kind: 'failed', error: 'download failed: disk full' });
    expect(view.getByRole('alert').textContent).toContain('disk full');
    fireEvent.click(view.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    fireEvent.click(view.getByRole('button', { name: 'Cancel download of Medium' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  test('an installed model can be deleted and says when it is in use', () => {
    const { view, onDelete } = show({ kind: 'installed' }, true);
    expect(view.container.textContent).toContain('Installed');
    expect(view.container.textContent).toContain('In use');
    fireEvent.click(view.getByRole('button', { name: 'Delete Medium' }));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });
});
