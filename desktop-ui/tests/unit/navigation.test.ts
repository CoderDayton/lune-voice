// SPDX-License-Identifier: GPL-3.0-or-later

import { setupI18n } from '@lingui/core';
import { describe, expect, test } from 'bun:test';
import { navigationGroups, navigationItems } from '../../src/features/shell/config/navigation';

describe('navigation', () => {
  test('every page title, subtitle and group name goes through the catalog', () => {
    const i18n = setupI18n({ locale: 'de', messages: { de: {} } });
    const texts = [
      ...navigationItems.flatMap((item) => [item.label, item.description]),
      ...navigationGroups.map((group) => group.name),
    ];
    for (const text of texts) {
      expect(typeof text).toBe('object');
      expect(i18n._(text).length).toBeGreaterThan(0);
    }
  });
});
