import { i18n } from "@lingui/core";

export const locales = {
  en: "English",
  es: "Español",
  fr: "Français",
  de: "Deutsch",
  pt: "Português",
  it: "Italiano",
  ja: "日本語",
  ko: "한국어",
  zh: "中文",
  ru: "Русский",
  ar: "العربية",
  hi: "हिन्दी",
} as const;

export type LocaleKey = keyof typeof locales;
export const defaultLocale: LocaleKey = "en";

export async function activateLocale(locale: string) {
  const { messages } = await import(`./locales/${locale}/messages.po`);
  i18n.load(locale, messages);
  i18n.activate(locale);
}
