// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Id of the DaisyUI drawer checkbox that drives the sidebar's expanded state.
 *
 * Lives here rather than in `app-shell` so the sidebar and its widgets can read
 * it without importing the shell that renders them — that import closed a cycle.
 */
export const DRAWER_ID = 'app-drawer';

/** Id of the sidebar's `<nav>`, a focus parking spot that activates nothing. */
export const SIDEBAR_NAV_ID = 'sidebar-nav';

/** The drawer checkbox is checked while the sidebar is expanded. */
export function isSidebarExpanded(): boolean {
  const toggle = document.getElementById(DRAWER_ID);
  return toggle instanceof HTMLInputElement ? toggle.checked : true;
}
