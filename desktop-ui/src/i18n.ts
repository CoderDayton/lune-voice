import { i18n } from "@lingui/core";

export const locales = {
  en: "English",
  es: "Español",
  de: "Deutsch",
} as const;

export type LocaleKey = keyof typeof locales;
export const defaultLocale: LocaleKey = "en";

export async function activateLocale(locale: string) {
  const { messages } = await import(`./locales/${locale}/messages.ts`);
  i18n.load(locale, messages);
  i18n.activate(locale);
}
