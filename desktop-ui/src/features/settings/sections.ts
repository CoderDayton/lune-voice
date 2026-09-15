// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The settings tab ids, shared by the page and the `/settings` route.
 *
 * Kept in its own module so the route can validate `?tab=` without importing
 * the (lazily loaded) page, and the page can type its state without importing
 * the route back.
 */
export const SETTINGS_SECTION_IDS = [
  'general',
  'appearance',
  'dictation',
  'pill',
  'transcription',
  'styling',
  'output',
  'models',
] as const;

export type SettingsSectionId = (typeof SETTINGS_SECTION_IDS)[number];

export function isSettingsSectionId(value: unknown): value is SettingsSectionId {
  return typeof value === 'string' && (SETTINGS_SECTION_IDS as readonly string[]).includes(value);
}
