// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import { SETTINGS_SECTION_IDS } from '../../src/features/settings/sections';
import { settingsRailGroups } from '../../src/features/settings/settings-rail';

describe('settingsRailGroups', () => {
  const ids = settingsRailGroups.flatMap((group) => group.items.map((item) => item.id));

  test('lists every settings section exactly once', () => {
    expect([...ids].sort()).toEqual([...SETTINGS_SECTION_IDS].sort());
  });

  test('opens on General, the default section', () => {
    expect(ids[0]).toBe('general');
  });

  test('puts Models beside Transcription so downloads are found where speech is set up', () => {
    const speech = settingsRailGroups.find((group) => group.items.some((i) => i.id === 'models'));
    expect(speech?.items.map((i) => i.id)).toContain('transcription');
  });

  test('gives every group and item a name', () => {
    for (const group of settingsRailGroups) {
      expect(group.name.id).toBeTruthy();
      for (const item of group.items) {
        expect(item.label.id).toBeTruthy();
      }
    }
  });
});
