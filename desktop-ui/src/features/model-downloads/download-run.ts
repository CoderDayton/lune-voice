// SPDX-License-Identifier: GPL-3.0-or-later

import { Store } from '@tanstack/store';
import { buildChecklist, type GpuStep, type RunChecklist, type RunModel } from './checklist';

/** One progress report for the model the backend is fetching. */
export interface DownloadEvent {
  modelName: string;
  /** The model the download was asked for, even while its VAD transfers. */
  localName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  /** Bytes fetched so far across the whole download, VAD included. */
  downloadedBytes: number;
  /** The download's size; 0 when unknown. */
  totalBytes: number;
  done: boolean;
}

/**
 * A run, written down so closing the window does not lose it.
 *
 * Closing the main window destroys the webview and everything held in it. The
 * transfer itself keeps going, since it runs on a blocking task in Rust, but
 * without this the models queued behind it were simply forgotten.
 */
export interface SavedRun {
  /** Models still to fetch, the one in flight first. */
  pending: string[];
  /** How many the run started with, so a progress count keeps its scale. */
  total: number;
  /** Opaque to the run: what the caller wants done once the models are in. */
  tag: string | null;
}

export interface DownloadDeps {
  download(localName: string): Promise<void>;
  cancel(): Promise<void>;
  /** Bytes a download of the model still has to fetch; `null` when unknown. */
  missingBytes(localName: string): Promise<number | null>;
  storage: {
    load(): SavedRun | null;
    save(run: SavedRun): void;
    clear(): void;
  };
}

export type RunStatus = 'idle' | 'running' | 'failed' | 'settled';

export interface RunState {
  status: RunStatus;
  /** Every model of the run, in queue order, with sizes fixed at the start. */
  models: RunModel[];
  /** Not started yet. */
  queue: string[];
  current: string | null;
  finished: string[];
  failed: string | null;
  error: string | null;
  /** Bytes of `current` fetched so far. */
  currentBytes: number;
  /** Bytes of a failed model that were on disk when it stopped. */
  kept: Record<string, number>;
  total: number;
  tag: string | null;
  /** Downloads finished since the app started; never reset with the run. */
  completions: number;
}

const IDLE: RunState = {
  status: 'idle',
  models: [],
  queue: [],
  current: null,
  finished: [],
  failed: null,
  error: null,
  currentBytes: 0,
  kept: {},
  total: 0,
  tag: null,
  completions: 0,
};

/** A model as the library knows it, for resuming a run. */
export interface LibraryModel extends RunModel {
  ready: boolean;
}

/**
 * The one download queue of the app, shared by the setup wizard and the Models
 * page: one model at a time, a count of bytes for the row in flight, a retry
 * that picks the failed model up again, and a cancel that the backend's late
 * answer cannot undo.
 *
 * It owns no UI and no I/O of its own: the transfer, the cancel and the parked
 * record come in as `deps`, and progress comes in through `onProgress`.
 */
export function createDownloadRun(deps: DownloadDeps) {
  const store = new Store<RunState>(IDLE);
  // Bumped whenever the run it belongs to ends, so a callback of a download
  // that was cancelled cannot report on a run started after it.
  let generation = 0;

  const patch = (next: Partial<RunState>) => store.setState((s) => ({ ...s, ...next }));
  const fresh = (next: Partial<RunState> = {}): RunState => ({
    ...IDLE,
    completions: store.state.completions,
    ...next,
  });

  function park() {
    const { current, queue, total, tag } = store.state;
    deps.storage.save({ pending: current ? [current, ...queue] : queue, total, tag });
  }

  function advance() {
    const [next, ...rest] = store.state.queue;
    if (next === undefined) {
      deps.storage.clear();
      patch({ current: null, currentBytes: 0, status: 'settled' });
      return;
    }
    patch({ queue: rest, current: next, currentBytes: 0 });
    // Written before the call, not after: the window can close at any moment
    // during the transfer this line is about to start.
    park();
    const mine = generation;
    deps.download(next).catch((err) => {
      // A cancel rejects the call by design; reporting that as a failed
      // download would show an error for the thing the user just asked for.
      if (mine !== generation) return;
      void fail(next, err, mine);
    });
  }

  async function fail(localName: string, err: unknown, mine: number) {
    const bytes = store.state.models.find((m) => m.localName === localName)?.bytes ?? 0;
    patch({
      failed: localName,
      error: String(err),
      current: null,
      currentBytes: 0,
      status: 'failed',
    });
    // Finished files survive a failure and a retry fetches only the rest, so
    // the row keeps counting them.
    const missing = await deps.missingBytes(localName).catch(() => null);
    if (mine !== generation || missing == null || bytes === 0) return;
    store.setState((s) => ({
      ...s,
      kept: { ...s.kept, [localName]: Math.max(0, bytes - missing) },
    }));
  }

  return {
    store,

    get state() {
      return store.state;
    },

    /**
     * Download `models`. During a run they join its queue, and the ones it
     * already has are left alone; otherwise they start a run of their own,
     * replacing a finished or failed one.
     */
    start(models: RunModel[], tag: string | null = null) {
      if (models.length === 0) return;
      const s = store.state;
      if (s.status === 'running') {
        const known = new Set(s.models.map((m) => m.localName));
        const added = models.filter((m) => !known.has(m.localName));
        if (added.length === 0) return;
        patch({
          models: [...s.models, ...added],
          queue: [...s.queue, ...added.map((m) => m.localName)],
          total: s.total + added.length,
          tag: s.tag ?? tag,
        });
        park();
        return;
      }
      generation += 1;
      store.setState(() =>
        fresh({
          status: 'running',
          models,
          queue: models.map((m) => m.localName),
          total: models.length,
          tag,
        }),
      );
      advance();
    },

    /** Stop the transfer in flight and forget the run. */
    cancel() {
      const inFlight = store.state.status === 'running';
      generation += 1;
      deps.storage.clear();
      store.setState(() => fresh());
      if (!inFlight) return;
      void deps.cancel().catch(() => {
        // The backend answers a cancel with nothing the user could act on.
      });
    },

    /** Put the failed model back at the head of the queue and carry on. */
    retry() {
      const s = store.state;
      if (s.status !== 'failed' || s.failed === null) return;
      patch({ queue: [s.failed, ...s.queue], failed: null, error: null, status: 'running' });
      advance();
    },

    /** Clear a run that has ended, so its checklist stops showing. */
    dismiss() {
      if (store.state.status === 'running' || store.state.status === 'idle') return;
      generation += 1;
      deps.storage.clear();
      store.setState(() => fresh());
    },

    /** Feed a progress report from the backend. */
    onProgress(event: DownloadEvent) {
      const s = store.state;
      if (s.status !== 'running' || event.localName !== s.current) return;
      if (!event.done) {
        patch({ currentBytes: event.downloadedBytes });
        return;
      }
      patch({
        finished: s.finished.includes(event.localName)
          ? s.finished
          : [...s.finished, event.localName],
        currentBytes: 0,
        completions: s.completions + 1,
      });
      advance();
    },

    /**
     * Pick up a run the last window lifetime left unfinished. The backend wins
     * where it disagrees with the parked record, which is written just before
     * each call: the model it is transferring goes first.
     *
     * Returns the run's tag when it took over, `null` when nothing resumed.
     */
    resume(library: LibraryModel[], active: string | null): { tag: string | null } | null {
      if (store.state.status === 'running') return null;
      const saved = deps.storage.load();
      if (!saved && !active) return null;

      const isReady = (name: string) => library.some((m) => m.localName === name && m.ready);
      const remaining = (saved?.pending ?? []).filter((name) => !isReady(name));
      const queue = active ? [active, ...remaining.filter((n) => n !== active)] : remaining;
      if (queue.length === 0) {
        deps.storage.clear();
        return null;
      }

      // Models finished before the window closed are not known by name, so the
      // resumed run lists only what is still owed. `total` only sets the scale
      // of a count, so a record that disagrees must not read "3 of 2".
      const models = queue.flatMap((name) => {
        const m = library.find((x) => x.localName === name);
        return m ? [{ localName: m.localName, name: m.name, bytes: m.bytes }] : [];
      });
      const tag = saved?.tag ?? null;
      generation += 1;
      store.setState(() =>
        fresh({
          status: 'running',
          models,
          queue,
          total: Math.max(saved?.total ?? queue.length, queue.length),
          tag,
        }),
      );
      // Re-issuing the command for a model already in flight is safe: the
      // backend recognises the duplicate and answers from the transfer already
      // running.
      advance();
      return { tag };
    },
  };
}

export type DownloadRun = ReturnType<typeof createDownloadRun>;

/** The run as a checklist, with the GPU plugin as an optional last row. */
export function runChecklist(
  run: RunState,
  gpu: { bytes: number; step: GpuStep; downloadedBytes: number } | null,
): RunChecklist {
  return buildChecklist({
    models: run.models,
    finished: new Set(run.finished),
    current: run.current,
    currentBytes: run.currentBytes,
    failed: run.failed,
    kept: new Map(Object.entries(run.kept)),
    gpu,
  });
}
