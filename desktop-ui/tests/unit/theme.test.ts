// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import {
  getThemeMode,
  normalizeThemeMode,
  setThemeMode,
  themeSchemeFor,
  toggledThemeMode,
} from '../../src/lib/theme';

describe('blocked storage', () => {
  test('a mode set this session is still the current mode', () => {
    const g = globalThis as Record<string, unknown>;
    const saved = { localStorage: g.localStorage, window: g.window, document: g.document };
    const boom = () => {
      throw new Error('blocked');
    };
    g.localStorage = { getItem: boom, setItem: boom };
    g.window = { matchMedia: () => ({ matches: false }) };
    g.document = { documentElement: { setAttribute: () => {} } };
    try {
      setThemeMode('light');
      expect(getThemeMode()).toBe('light');
    } finally {
      Object.assign(g, saved);
    }
  });
});

describe('normalizeThemeMode', () => {
  test('a fresh install is dark', () => {
    expect(normalizeThemeMode(null)).toBe('dark');
  });

  test('current values pass through', () => {
    expect(normalizeThemeMode('dark')).toBe('dark');
    expect(normalizeThemeMode('light')).toBe('light');
    expect(normalizeThemeMode('system')).toBe('system');
  });

  test('the old Lune theme names map to their side', () => {
    expect(normalizeThemeMode('lune')).toBe('dark');
    expect(normalizeThemeMode('lune-light')).toBe('light');
  });

  test('a retired stock theme maps to its light or dark side', () => {
    expect(normalizeThemeMode('cupcake')).toBe('light');
    expect(normalizeThemeMode('nord')).toBe('light');
    expect(normalizeThemeMode('dracula')).toBe('dark');
    expect(normalizeThemeMode('forest')).toBe('dark');
  });

  test('an unknown value falls back to dark', () => {
    expect(normalizeThemeMode('nonsense')).toBe('dark');
  });
});

describe('themeSchemeFor', () => {
  test('system follows the OS', () => {
    expect(themeSchemeFor('system', true)).toBe('light');
    expect(themeSchemeFor('system', false)).toBe('dark');
  });

  test('an explicit mode ignores the OS', () => {
    expect(themeSchemeFor('dark', true)).toBe('dark');
    expect(themeSchemeFor('light', false)).toBe('light');
  });
});

describe('toggledThemeMode', () => {
  test('flips an explicit mode', () => {
    expect(toggledThemeMode('dark', false)).toBe('light');
    expect(toggledThemeMode('light', true)).toBe('dark');
  });

  test('flips the look system currently shows', () => {
    expect(toggledThemeMode('system', true)).toBe('dark');
    expect(toggledThemeMode('system', false)).toBe('light');
  });
});
