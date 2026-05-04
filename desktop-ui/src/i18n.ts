import { i18n } from '@lingui/core';

// Locales shown in the language selector. Only include languages with >80%
// translation coverage. .po files for other languages are kept for ongoing
// translation work — add them here once they cross the threshold.
export const locales = {
  en: 'English',
} as const;

export type LocaleKey = keyof typeof locales;
export const defaultLocale: LocaleKey = 'en';

export async function activateLocale(locale: string) {
  const { messages } = await import(`./locales/${locale}/messages.po`);
  i18n.load(locale, messages);
  i18n.activate(locale);
}
