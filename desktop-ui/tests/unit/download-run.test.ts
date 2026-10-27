// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import {
  createDownloadRun,
  type DownloadDeps,
  type DownloadEvent,
  runChecklist,
  type SavedRun,
} from '../../src/features/model-downloads/download-run';

const MB = 1024 * 1024;

const model = (localName: string, bytes = 100 * MB) => ({ localName, name: localName, bytes });

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup(opts: { missing?: Record<string, number | null>; saved?: SavedRun | null } = {}) {
  const calls: string[] = [];
  const pending = new Map<string, { resolve: () => void; reject: (e: unknown) => void }>();
  let saved: SavedRun | null = opts.saved ?? null;
  let cancels = 0;
  const deps: DownloadDeps = {
    download: (name) => {
      calls.push(name);
      return new Promise<void>((resolve, reject) => pending.set(name, { resolve, reject }));
    },
    cancel: async () => {
      cancels += 1;
    },
    missingBytes: async (name) => opts.missing?.[name] ?? null,
    storage: {
      load: () => saved,
      save: (run) => {
        saved = run;
      },
      clear: () => {
        saved = null;
      },
    },
  };
  const run = createDownloadRun(deps);
  const done = (localName: string): DownloadEvent => ({
    modelName: localName,
    localName,
    fileName: 'model.bin',
    fileIndex: 0,
    fileCount: 1,
    downloadedBytes: 100 * MB,
    totalBytes: 100 * MB,
    done: true,
  });
  return { run, calls, pending, done, saved: () => saved, cancels: () => cancels };
}

describe('createDownloadRun', () => {
  test('downloads the models one after another and settles after the last', () => {
    const t = setup();
    t.run.start([model('a'), model('b')]);
    expect(t.run.state.status).toBe('running');
    expect(t.calls).toEqual(['a']);

    t.run.onProgress(t.done('a'));
    expect(t.calls).toEqual(['a', 'b']);
    expect(t.run.state.finished).toEqual(['a']);

    t.run.onProgress(t.done('b'));
    expect(t.run.state.status).toBe('settled');
    expect(t.run.state.current).toBeNull();
    expect(t.run.state.finished).toEqual(['a', 'b']);
  });

  test('parks what is still owed before each download and clears it at the end', () => {
    const t = setup();
    t.run.start([model('a'), model('b')], 'gpu');
    expect(t.saved()).toEqual({ pending: ['a', 'b'], total: 2, tag: 'gpu' });

    t.run.onProgress(t.done('a'));
    expect(t.saved()).toEqual({ pending: ['b'], total: 2, tag: 'gpu' });

    t.run.onProgress(t.done('b'));
    expect(t.saved()).toBeNull();
  });

  test('tracks the bytes of the model in flight only', () => {
    const t = setup();
    t.run.start([model('a'), model('b')]);
    t.run.onProgress({ ...t.done('b'), done: false, downloadedBytes: 5 * MB });
    expect(t.run.state.currentBytes).toBe(0);
    t.run.onProgress({ ...t.done('a'), done: false, downloadedBytes: 40 * MB });
    expect(t.run.state.currentBytes).toBe(40 * MB);
  });

  test('a failed download stops the run, and retry picks the same model up again', async () => {
    const t = setup({ missing: { a: 30 * MB } });
    t.run.start([model('a'), model('b')]);
    t.pending.get('a')?.reject('network down');
    await flush();

    expect(t.run.state.status).toBe('failed');
    expect(t.run.state.failed).toBe('a');
    expect(t.run.state.error).toBe('network down');
    expect(t.run.state.kept.a).toBe(70 * MB);

    t.run.retry();
    expect(t.run.state.status).toBe('running');
    expect(t.calls).toEqual(['a', 'a']);

    t.run.onProgress(t.done('a'));
    expect(t.calls).toEqual(['a', 'a', 'b']);
  });

  test('cancel ends the run, and what the backend answers afterwards is ignored', async () => {
    const t = setup();
    t.run.start([model('a'), model('b')]);
    t.run.cancel();
    t.pending.get('a')?.reject('cancelled');
    await flush();

    expect(t.run.state.status).toBe('idle');
    expect(t.run.state.error).toBeNull();
    expect(t.cancels()).toBe(1);
    expect(t.saved()).toBeNull();

    t.run.onProgress(t.done('a'));
    expect(t.calls).toEqual(['a']);
    expect(t.run.state.status).toBe('idle');
  });

  test('models added during a run join its queue without restarting it', () => {
    const t = setup();
    t.run.start([model('a')]);
    t.run.start([model('a'), model('b')]);
    expect(t.calls).toEqual(['a']);
    expect(t.run.state.models.map((m) => m.localName)).toEqual(['a', 'b']);
    expect(t.run.state.total).toBe(2);

    t.run.onProgress(t.done('a'));
    expect(t.calls).toEqual(['a', 'b']);
  });

  test('a tag given to a joining start is kept, and an existing tag is not overwritten', () => {
    const t = setup();
    t.run.start([model('a')]);
    t.run.start([model('b')], 'gpu');
    expect(t.run.state.tag).toBe('gpu');
    t.run.start([model('c')], 'other');
    expect(t.run.state.tag).toBe('gpu');
    expect(t.saved()?.tag).toBe('gpu');
  });

  test('cancel only tells the backend while a transfer is in flight', async () => {
    const t = setup();
    t.run.start([model('a')]);
    t.pending.get('a')?.reject('network down');
    await flush();
    expect(t.run.state.status).toBe('failed');
    t.run.cancel();
    expect(t.run.state.status).toBe('idle');
    expect(t.cancels()).toBe(0);

    t.run.cancel();
    expect(t.cancels()).toBe(0);
  });

  test('the completion count only grows, through replaced and dismissed runs', () => {
    const t = setup();
    t.run.start([model('a'), model('b')]);
    t.run.onProgress(t.done('a'));
    expect(t.run.state.completions).toBe(1);
    t.run.onProgress(t.done('b'));
    expect(t.run.state.completions).toBe(2);

    t.run.dismiss();
    expect(t.run.state.completions).toBe(2);
    t.run.start([model('c')]);
    expect(t.run.state.completions).toBe(2);
    t.run.onProgress(t.done('c'));
    expect(t.run.state.completions).toBe(3);
  });

  test('a new run replaces a finished one', () => {
    const t = setup();
    t.run.start([model('a')]);
    t.run.onProgress(t.done('a'));
    t.run.start([model('b')]);
    expect(t.run.state.models.map((m) => m.localName)).toEqual(['b']);
    expect(t.run.state.finished).toEqual([]);
    expect(t.calls).toEqual(['a', 'b']);
  });

  test('dismiss clears a finished run', () => {
    const t = setup();
    t.run.start([model('a')]);
    t.run.onProgress(t.done('a'));
    t.run.dismiss();
    expect(t.run.state.status).toBe('idle');
    expect(t.run.state.models).toEqual([]);
  });

  test('dismiss leaves a run that is still going alone', () => {
    const t = setup();
    t.run.start([model('a')]);
    t.run.dismiss();
    expect(t.run.state.status).toBe('running');
  });

  describe('resume', () => {
    const library = [
      { ...model('a'), ready: false },
      { ...model('b'), ready: true },
      { ...model('c'), ready: false },
    ];

    test('takes up the parked queue, minus what is already installed, with the live one first', () => {
      const t = setup({ saved: { pending: ['a', 'b', 'c'], total: 3, tag: 'gpu' } });
      const resumed = t.run.resume(library, 'c');
      expect(resumed).toEqual({ tag: 'gpu' });
      expect(t.calls).toEqual(['c']);
      expect(t.run.state.models.map((m) => m.localName)).toEqual(['c', 'a']);
      expect(t.run.state.total).toBe(3);
    });

    test('reports nothing to resume when nothing is owed', () => {
      const t = setup({ saved: { pending: ['b'], total: 1, tag: null } });
      expect(t.run.resume(library, null)).toBeNull();
      expect(t.saved()).toBeNull();
      expect(t.calls).toEqual([]);
    });

    test('does nothing when a run is already going', () => {
      const t = setup({ saved: { pending: ['a'], total: 1, tag: null } });
      t.run.start([model('c')]);
      expect(t.run.resume(library, null)).toBeNull();
      expect(t.calls).toEqual(['c']);
    });
  });
});

describe('runChecklist', () => {
  test('counts the models of the run and the bytes of the one in flight', () => {
    const t = setup();
    t.run.start([model('a'), model('b')]);
    t.run.onProgress({ ...t.done('a'), done: false, downloadedBytes: 25 * MB });
    const list = runChecklist(t.run.state, null);
    expect(list.rows.map((r) => [r.id, r.state])).toEqual([
      ['a', 'active'],
      ['b', 'queued'],
    ]);
    expect(list.doneBytes).toBe(25 * MB);
    expect(list.totalBytes).toBe(200 * MB);
  });
});
