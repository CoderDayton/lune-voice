import type { LinguiConfig } from "@lingui/conf";

const config: LinguiConfig = {
  sourceLocale: "en",
  locales: ["en", "es", "de"],
  catalogs: [
    {
      path: "src/locales/{locale}/messages",
      include: ["src"],
    },
  ],
};

export default config;
