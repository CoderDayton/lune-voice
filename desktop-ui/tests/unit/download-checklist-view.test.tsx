// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, render } from '@testing-library/react';
import { DownloadChecklist } from '../../src/features/welcome/components/download-checklist';
import { buildChecklist } from '../../src/features/welcome/setup-logic';

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

function show(gpuStep: 'queued' | 'activating' | 'done', gpuActive = false) {
  const checklist = buildChecklist({
    models: [{ localName: 'whisper-medium', name: 'Whisper Medium', bytes: 540 * MB }],
    finished: new Set(gpuStep === 'queued' ? [] : ['whisper-medium']),
    current: gpuStep === 'queued' ? 'whisper-medium' : null,
    currentBytes: 210 * MB,
    failed: null,
    gpu: { bytes: 17 * MB, step: gpuStep, downloadedBytes: gpuStep === 'queued' ? 0 : 17 * MB },
  });
  return render(
    <I18nProvider i18n={i18n}>
      <DownloadChecklist checklist={checklist} gpuActive={gpuActive} gpuError={null} />
    </I18nProvider>,
  );
}

describe('DownloadChecklist', () => {
  test('shows the active model with a byte bar and the GPU waiting behind it', () => {
    const view = show('queued');
    const rows = view.getAllByRole('listitem');
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute('aria-current')).toBe('step');
    expect(rows[0].textContent).toContain('Whisper Medium');
    expect(rows[0].textContent).toContain('210');
    expect(rows[1].textContent).toContain('GPU support');
    expect(rows[1].textContent).toContain('Waiting');
    expect(view.container.textContent).toContain('38%');
  });

  test('says the GPU is being turned on after its download', () => {
    const rows = show('activating').getAllByRole('listitem');
    expect(rows[0].textContent).toContain('Downloaded');
    expect(rows[1].textContent).toContain('Turning on your GPU…');
  });

  test('says transcription runs on the GPU once it is live', () => {
    const rows = show('done', true).getAllByRole('listitem');
    expect(rows[1].textContent).toContain('Transcription now runs on your GPU.');
  });

  test('drops the overall bar once everything is settled', () => {
    expect(show('queued').container.textContent).toContain('Overall progress');
    cleanup();
    expect(show('done', true).container.textContent).not.toContain('Overall progress');
  });

  test('leaves the failure message to the alert below the list', () => {
    const checklist = buildChecklist({
      models: [{ localName: 'whisper-medium', name: 'Whisper Medium', bytes: 540 * MB }],
      finished: new Set(),
      current: null,
      currentBytes: 0,
      failed: 'whisper-medium',
      gpu: null,
    });
    const view = render(
      <I18nProvider i18n={i18n}>
        <DownloadChecklist checklist={checklist} gpuActive={false} gpuError={null} />
      </I18nProvider>,
    );
    const row = view.getAllByRole('listitem')[0];
    expect(row.textContent).not.toContain('Download failed');
    expect(row.textContent).toContain('Interrupted');
  });
});
