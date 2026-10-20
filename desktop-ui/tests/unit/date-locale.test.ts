// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import { dateLocale } from '../../src/i18n';

describe('dateLocale', () => {
  test('follows the app language', () => {
    expect(dateLocale('de', ['en-US'])).toBe('de');
  });

  test('keeps the system region when it speaks the same language', () => {
    expect(dateLocale('en', ['en-GB', 'de-DE'])).toBe('en-GB');
    expect(dateLocale('pt', ['de-DE', 'pt-BR'])).toBe('pt-BR');
  });
});
