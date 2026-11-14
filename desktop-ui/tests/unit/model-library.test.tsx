// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { modelDownloads } from '../../src/features/model-downloads/model-downloads';
import { ModelLibrary } from '../../src/features/settings/components/model-library';
import { whisperModels } from '../../src/features/settings/data/settings-data';
import type { ModelStatus, ModelStatusApi } from '../../src/features/settings/hooks/use-model-status';

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

function entry(localName: string, over: Partial<ModelStatus> = {}): ModelStatus {
  return {
    name: localName,
    repoId: 'x/y',
    localName,
    ready: false,
    description: '',
    required: false,
    inUse: false,
    role: 'speech-to-text',
    sizeBytes: null,
    downloadBytes: 100 * MB,
    ...over,
  };
}

function show(models: ModelStatus[], diskBytes: number | null = null) {
  const status: ModelStatusApi = {
    models,
    byName: new Map(models.map((m) => [m.localName, m])),
    diskBytes,
    // Loading keeps the resume-on-mount call to the backend from running.
    loading: true,
    refresh: () => {},
    remove: async () => {},
  };
  return render(
    <I18nProvider i18n={i18n}>
      <ModelLibrary status={status} />
    </I18nProvider>,
  );
}

const rowNames = (view: ReturnType<typeof show>) =>
  view.getAllByRole('button', { name: /^(Download|Delete) / }).map((b) => b.getAttribute('aria-label'));

describe('ModelLibrary', () => {
  test('shows one family at a time, starting with the one in use', () => {
    const view = show([
      entry('parakeet-tdt-0.6b-v3-q8_0', { ready: true, inUse: true, sizeBytes: 669 * MB }),
    ]);
    const tabs = view.getAllByRole('tab');
    expect(tabs.map((t) => t.textContent?.trim())).toEqual(['Whisper', 'Parakeet']);
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(rowNames(view)).toEqual(['Download Compact', 'Delete Full']);
  });

  test('switching tabs swaps the list instead of growing the page', () => {
    const view = show([]);
    expect(rowNames(view)).toHaveLength(4);
    fireEvent.click(view.getByRole('tab', { name: 'Parakeet' }));
    expect(rowNames(view)).toEqual(['Download Compact', 'Download Full']);
    expect(view.container.textContent).not.toContain('Large v3 Turbo');
  });

  test('has no multi-select bar', () => {
    const view = show([]);
    expect(view.queryByRole('checkbox')).toBeNull();
    expect(view.queryByRole('button', { name: 'Select all' })).toBeNull();
  });

  test('a download during a failed run keeps the models queued behind the failure', () => {
    const [a, b] = whisperModels;
    const run = (m: (typeof whisperModels)[number]) => ({
      localName: m.localName,
      name: m.localName,
      bytes: MB,
    });
    modelDownloads.store.setState((s) => ({
      ...s,
      status: 'failed',
      models: [run(a), run(b)],
      queue: [b.localName],
      current: null,
      failed: a.localName,
      error: 'disk full',
      total: 2,
    }));
    const view = show([]);
    fireEvent.click(view.getAllByRole('button', { name: /^Download / })[0]);
    const { current, queue } = modelDownloads.state;
    modelDownloads.cancel();
    expect(current).toBe(b.localName);
    expect(queue).toEqual([whisperModels[2].localName]);
  });

  test('disk usage sits beside the tabs', () => {
    const view = show([
      entry('whisper-base-q8_0', { ready: true, sizeBytes: 78 * MB }),
    ]);
    expect(view.getByText('78 MB used')).toBeTruthy();
  });

  test('disk usage takes the backend figure, which counts the hidden VAD', () => {
    const view = show([entry('whisper-base-q8_0', { ready: true, sizeBytes: 78 * MB })], 80 * MB);
    expect(view.getByText('80 MB used')).toBeTruthy();
  });
});
