// SPDX-License-Identifier: GPL-3.0-or-later

import type { LinguiConfig } from "@lingui/conf";

const config: LinguiConfig = {
  sourceLocale: "en",
  locales: ["en", "es", "fr", "de", "pt", "it", "ja", "ko", "zh", "ru", "ar", "hi"],
  catalogs: [
    {
      path: "src/locales/{locale}/messages",
      include: ["src"],
    },
  ],
};

export default config;
