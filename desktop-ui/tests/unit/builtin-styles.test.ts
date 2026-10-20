// SPDX-License-Identifier: GPL-3.0-or-later

import { setupI18n } from '@lingui/core';
import { describe, expect, test } from 'bun:test';
import {
  presetDescription,
  presetName,
  styleLabel,
} from '../../src/features/styles/builtin-styles';

const i18n = setupI18n({ locale: 'en', messages: { en: {} } });
i18n.activate('en');

describe('builtin styles', () => {
  test('a built-in preset is named from the catalog, not the backend text', () => {
    const casual = { id: 'casual', name: 'ignored', description: 'ignored', isBuiltin: true };
    expect(presetName(i18n, casual)).toBe('Casual');
    expect(presetDescription(i18n, casual)).toBe('Texting, messaging, chat');
  });

  test('your own preset keeps the text you gave it', () => {
    const mine = { id: 'casual', name: 'Mine', description: 'My own', isBuiltin: false };
    expect(presetName(i18n, mine)).toBe('Mine');
    expect(presetDescription(i18n, mine)).toBe('My own');
  });

  test('a session style id resolves to a built-in name or the matching custom preset', () => {
    const custom = [{ id: 'u-1', name: 'Pirate', description: '', isBuiltin: false }];
    expect(styleLabel(i18n, 'technical', custom)).toBe('Technical');
    expect(styleLabel(i18n, 'u-1', custom)).toBe('Pirate');
    expect(styleLabel(i18n, 'gone', custom)).toBe('gone');
  });
});
