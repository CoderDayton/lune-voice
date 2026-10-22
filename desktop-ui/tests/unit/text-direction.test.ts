// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import { locales, textDirection } from '../../src/i18n';

describe('textDirection', () => {
  test('Arabic reads right to left', () => {
    expect(textDirection('ar')).toBe('rtl');
  });

  test('every other shipped language reads left to right', () => {
    for (const locale of Object.keys(locales).filter((l) => l !== 'ar')) {
      expect(textDirection(locale)).toBe('ltr');
    }
  });
});
