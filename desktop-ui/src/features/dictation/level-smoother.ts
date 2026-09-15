// SPDX-License-Identifier: GPL-3.0-or-later

/** Fast-attack / slow-release exponential smoothing factors. */
const ATTACK_ALPHA = 0.35;
const RELEASE_ALPHA = 0.15;

export interface FrameScheduler {
  request(cb: () => void): number;
  cancel(id: number): void;
}

const rafScheduler: FrameScheduler = {
  request: (cb) => requestAnimationFrame(cb),
  cancel: (id) => cancelAnimationFrame(id),
};

/**
 * Smooths raw audio-level events and coalesces them into at most one commit
 * per animation frame.
 *
 * The backend emits levels at ~15 Hz, but nothing bounds what arrives on the
 * event channel; committing per event would do style/paint work the display
 * can't show — WebKitGTK repaints the whole transparent pill overlay per
 * update, which is what made the Linux pill stutter. Every push advances the
 * smoothed value; a single scheduled frame commits the latest.
 */
export class LevelSmoother {
  private smoothed = 0;
  private frame: number | null = null;

  constructor(
    private readonly commit: (level: number) => void,
    private readonly scheduler: FrameScheduler = rafScheduler,
  ) {}

  /** Current smoothed level (committed or pending). */
  get value(): number {
    return this.smoothed;
  }

  push(level: number): void {
    const alpha = level > this.smoothed ? ATTACK_ALPHA : RELEASE_ALPHA;
    this.smoothed += alpha * (level - this.smoothed);
    this.frame ??= this.scheduler.request(() => {
      this.frame = null;
      this.commit(this.smoothed);
    });
  }

  /** Drop any pending frame without committing it. */
  cancelPending(): void {
    if (this.frame !== null) {
      this.scheduler.cancel(this.frame);
      this.frame = null;
    }
  }

  /** Cancel pending work and return to silence. Does not call commit. */
  reset(): void {
    this.cancelPending();
    this.smoothed = 0;
  }
}
