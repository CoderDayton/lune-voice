import type { PillMode } from './use-pill-state';

/**
 * How long `processing` must persist before the pill shows it, and how long it
 * stays once shown. Mirrors `PROCESSING_DEBUT` / `PROCESSING_DWELL` in
 * pill_native.rs.
 */
const PROCESSING_DEBUT_MS = 150;
const PROCESSING_DWELL_MS = 250;

export interface Clock {
  now(): number;
  setTimeout(cb: () => void, ms: number): number;
  clearTimeout(id: number): void;
}

const wallClock: Clock = {
  now: () => Date.now(),
  setTimeout: (cb, ms) => window.setTimeout(cb, ms),
  clearTimeout: (id) => window.clearTimeout(id),
};

/**
 * Decides which mode the pill actually renders, which is not always the one the
 * runtime last published.
 *
 * An utterance with no speech in it is trimmed to nothing by the VAD and decodes
 * instantly, so the backend can publish `processing` and withdraw it inside a
 * single frame. Rendering that is a flash of a different hue carrying no
 * information, which reads as a glitch rather than as a state. So a state too
 * brief to be read is never drawn, and one worth drawing stays long enough to
 * see.
 *
 * Only `processing` is gated: `active` is the feedback for the hotkey and has to
 * be immediate.
 */
export class ModeGate {
  private shown: PillMode = 'idle';
  private shownAt = 0;
  private timer: number | null = null;

  constructor(
    private readonly commit: (mode: PillMode) => void,
    private readonly clock: Clock = wallClock,
  ) {}

  /** Current rendered mode. */
  get value(): PillMode {
    return this.shown;
  }

  push(mode: PillMode): void {
    this.cancelPending();
    if (mode === this.shown) return;

    if (mode === 'processing') {
      this.after(PROCESSING_DEBUT_MS, mode);
      return;
    }
    // Leaving a processing state it has not held long enough to be read.
    if (this.shown === 'processing') {
      const held = this.clock.now() - this.shownAt;
      if (held < PROCESSING_DWELL_MS) {
        this.after(PROCESSING_DWELL_MS - held, mode);
        return;
      }
    }
    this.show(mode);
  }

  /** Drop any withheld mode change. */
  cancelPending(): void {
    if (this.timer !== null) {
      this.clock.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private after(ms: number, mode: PillMode): void {
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      this.show(mode);
    }, ms);
  }

  private show(mode: PillMode): void {
    this.shown = mode;
    this.shownAt = this.clock.now();
    this.commit(mode);
  }
}
