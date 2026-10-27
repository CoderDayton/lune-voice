// SPDX-License-Identifier: GPL-3.0-or-later

/** A model in a download run, with its size fixed when the run started. */
export interface RunModel {
  localName: string;
  name: string;
  bytes: number;
}

export type GpuStep = 'queued' | 'downloading' | 'activating' | 'done' | 'failed';

export type ChecklistState = 'queued' | 'active' | 'activating' | 'done' | 'failed';

export interface ChecklistRow {
  /** The model's local name, or `gpu`. */
  id: string;
  /** The model's name; empty for the GPU row, which the UI labels. */
  name: string;
  state: ChecklistState;
  doneBytes: number;
  totalBytes: number;
}

export interface ChecklistInput {
  models: RunModel[];
  finished: ReadonlySet<string>;
  current: string | null;
  /** Bytes of `current` fetched so far. */
  currentBytes: number;
  failed: string | null;
  /**
   * Bytes of a model already on disk when its download stopped. Downloads
   * resume per file, so finished files stay and a retry fetches only the rest.
   */
  kept?: ReadonlyMap<string, number>;
  gpu: { bytes: number; step: GpuStep; downloadedBytes: number } | null;
}

/** Every download of a run, for the checklist. */
export interface RunChecklist {
  rows: ChecklistRow[];
  doneBytes: number;
  totalBytes: number;
  /** No row is waiting or working. */
  settled: boolean;
}

/**
 * Every download in a run as one ordered list: the models in queue order,
 * then GPU support. The run downloads one item at a time, so at most one row is
 * active.
 */
export function buildChecklist(input: ChecklistInput): RunChecklist {
  const rows: ChecklistRow[] = input.models.map((m) => {
    const state: ChecklistState = input.finished.has(m.localName)
      ? 'done'
      : m.localName === input.failed
        ? 'failed'
        : m.localName === input.current
          ? 'active'
          : 'queued';
    const kept = input.kept?.get(m.localName) ?? 0;
    const doneBytes =
      state === 'done'
        ? m.bytes
        : state === 'active'
          ? Math.min(kept + input.currentBytes, m.bytes)
          : state === 'failed'
            ? Math.min(kept, m.bytes)
            : 0;
    return { id: m.localName, name: m.name, state, doneBytes, totalBytes: m.bytes };
  });

  if (input.gpu) {
    const { bytes, step, downloadedBytes } = input.gpu;
    const state: ChecklistState = step === 'downloading' ? 'active' : step;
    const doneBytes =
      step === 'activating' || step === 'done'
        ? bytes
        : step === 'downloading'
          ? Math.min(downloadedBytes, bytes)
          : 0;
    rows.push({ id: 'gpu', name: '', state, doneBytes, totalBytes: bytes });
  }

  // A failed GPU install is not retried by setup, so its bytes are owed by
  // nothing and would leave a finished setup short of 100%.
  const counted = rows.filter((r) => !(r.id === 'gpu' && r.state === 'failed'));
  return {
    rows,
    doneBytes: counted.reduce((sum, r) => sum + r.doneBytes, 0),
    totalBytes: counted.reduce((sum, r) => sum + r.totalBytes, 0),
    settled: rows.every((r) => r.state === 'done' || r.state === 'failed'),
  };
}
