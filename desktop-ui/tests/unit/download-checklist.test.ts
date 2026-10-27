// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'bun:test';
import { buildChecklist, setupDownloadBytes } from '../../src/features/welcome/setup-logic';

const models = [
  { localName: 'a', name: 'A', bytes: 100 },
  { localName: 'b', name: 'B', bytes: 200 },
  { localName: 'c', name: 'C', bytes: 300 },
];

describe('buildChecklist', () => {
  it('marks each model done, active or queued and sums the bytes', () => {
    const list = buildChecklist({
      models,
      finished: new Set(['a']),
      current: 'b',
      currentBytes: 50,
      failed: null,
      gpu: { bytes: 17, step: 'queued', downloadedBytes: 0 },
    });
    expect(list.rows.map((r) => [r.id, r.state, r.doneBytes])).toEqual([
      ['a', 'done', 100],
      ['b', 'active', 50],
      ['c', 'queued', 0],
      ['gpu', 'queued', 0],
    ]);
    expect(list.totalBytes).toBe(617);
    expect(list.doneBytes).toBe(150);
  });

  it('shows the model that failed as failed', () => {
    const list = buildChecklist({
      models,
      finished: new Set(),
      current: null,
      currentBytes: 0,
      failed: 'a',
      gpu: null,
    });
    expect(list.rows.map((r) => r.state)).toEqual(['failed', 'queued', 'queued']);
  });

  it('counts the GPU download, then all of it while the GPU is being turned on', () => {
    const base = { models: [], finished: new Set<string>(), current: null, currentBytes: 0, failed: null };
    const downloading = buildChecklist({
      ...base,
      gpu: { bytes: 17, step: 'downloading', downloadedBytes: 4 },
    });
    expect(downloading.rows[0]).toMatchObject({ id: 'gpu', state: 'active', doneBytes: 4 });

    const activating = buildChecklist({
      ...base,
      gpu: { bytes: 17, step: 'activating', downloadedBytes: 17 },
    });
    expect(activating.rows[0]).toMatchObject({ state: 'activating', doneBytes: 17 });
    expect(activating.doneBytes).toBe(17);
  });

  it('drops a failed GPU install from the overall bar, so a finished setup reads 100%', () => {
    const list = buildChecklist({
      models: [models[0]],
      finished: new Set(['a']),
      current: null,
      currentBytes: 0,
      failed: null,
      gpu: { bytes: 17, step: 'failed', downloadedBytes: 17 },
    });
    expect(list.rows[1]).toMatchObject({ id: 'gpu', state: 'failed', totalBytes: 17 });
    expect(list.totalBytes).toBe(100);
    expect(list.doneBytes).toBe(100);
  });

  it('counts the files a failed model kept, and adds them back while it retries', () => {
    const base = { models: [models[1]], finished: new Set<string>(), gpu: null };
    const kept = new Map([['b', 60]]);
    const failed = buildChecklist({ ...base, current: null, currentBytes: 0, failed: 'b', kept });
    expect(failed.rows[0]).toMatchObject({ state: 'failed', doneBytes: 60 });
    const retrying = buildChecklist({ ...base, current: 'b', currentBytes: 30, failed: null, kept });
    expect(retrying.rows[0]).toMatchObject({ state: 'active', doneBytes: 90 });
  });

  it('is settled only once no row is waiting or working', () => {
    const base = { models: [models[0]], current: null, currentBytes: 0, failed: null };
    const working = buildChecklist({
      ...base,
      finished: new Set(['a']),
      gpu: { bytes: 17, step: 'activating', downloadedBytes: 17 },
    });
    expect(working.settled).toBe(false);
    const settled = buildChecklist({
      ...base,
      finished: new Set(['a']),
      gpu: { bytes: 17, step: 'failed', downloadedBytes: 0 },
    });
    expect(settled.settled).toBe(true);
  });

  it('never counts more bytes for a row than its size', () => {
    const list = buildChecklist({
      models: [models[0]],
      finished: new Set(),
      current: 'a',
      currentBytes: 150,
      failed: null,
      gpu: null,
    });
    expect(list.rows[0].doneBytes).toBe(100);
    expect(list.doneBytes).toBe(100);
  });
});

describe('setupDownloadBytes', () => {
  const status = (localName: string, ready: boolean, downloadBytes: number | null) => ({
    name: localName,
    repoId: 'r',
    localName,
    ready,
    description: '',
    required: false,
    inUse: true,
    downloadBytes,
  });

  it('adds the selected missing models and the GPU download', () => {
    const list = [status('a', false, 100), status('b', true, 0), status('c', false, 300)];
    expect(setupDownloadBytes(list, new Set(['a', 'b']), 17)).toBe(117);
    expect(setupDownloadBytes(list, new Set(['a', 'c']), 0)).toBe(400);
  });

  it('is unknown when a selected model has no known size', () => {
    expect(setupDownloadBytes([status('a', false, null)], new Set(['a']), 17)).toBeNull();
  });
});
