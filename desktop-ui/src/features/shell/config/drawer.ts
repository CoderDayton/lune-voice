/**
 * Id of the DaisyUI drawer checkbox that drives the sidebar's expanded state.
 *
 * Lives here rather than in `app-shell` so the sidebar and its widgets can read
 * it without importing the shell that renders them — that import closed a cycle.
 */
export const DRAWER_ID = 'app-drawer';
