// SPDX-License-Identifier: GPL-3.0-or-later

import { i18n } from '@lingui/core';

// Locales shown in the language selector. Every entry must have a translated
// catalog in src/locales/<key>/messages.po; untranslated strings fall back to
// the English source text.
export const locales = {
  en: 'English',
  ar: 'العربية',
  de: 'Deutsch',
  es: 'Español',
  fr: 'Français',
  hi: 'हिन्दी',
  it: 'Italiano',
  ja: '日本語',
  ko: '한국어',
  pt: 'Português',
  ru: 'Русский',
  zh: '中文',
} as const;

export type LocaleKey = keyof typeof locales;
export const defaultLocale: LocaleKey = 'en';

export async function activateLocale(locale: string) {
  const { messages } = await import(`./locales/${locale}/messages.po`);
  i18n.load(locale, messages);
  i18n.activate(locale);
}
