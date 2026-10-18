// SPDX-License-Identifier: GPL-3.0-or-later

import { setupI18n } from "@lingui/core";
import { describe, expect, test } from "bun:test";
import {
  activationModes,
  asrBackends,
  deliveryMethods,
  dictationLanguages,
  dictationLanguagesFor,
  engineAcceptsLanguage,
  engineModelLocalName,
  formatLanguageList,
  historyRetentionDays,
  historyRetentionOptions,
  historyRetentionValue,
  labelFor,
  languageForEngine,
  localizeText,
  modelDownloadSizes,
  parakeetLanguages,
  parakeetModels,
  parakeetSupportsLanguage,
  pillVisibilityOptions,
  whisperModels,
} from "../../src/features/settings/data/settings-data";

// No catalog loaded, so every descriptor resolves to its own source message —
// which is what these assertions are about.
const i18n = setupI18n({ locale: "en", messages: { en: {} } });

describe("settings data", () => {
  test("activationModes has Toggle and Hold", () => {
    const values = activationModes.map((o) => o.value);
    expect(values).toContain("Toggle");
    expect(values).toContain("Hold");
  });

  test("asrBackends has WhisperCpp, ParakeetCpp and OpenAiCompatible", () => {
    const values = asrBackends.map((o) => o.value);
    expect(values).toContain("WhisperCpp");
    expect(values).toContain("ParakeetCpp");
    expect(values).toContain("OpenAiCompatible");
    expect(values).not.toContain("Qwen3Asr");
  });

  test("Parakeet's model directories match the backend's", () => {
    // `lune_core::models::DIR_PARAKEET_TDT_V3_Q4K` / `_Q8`, smallest first.
    expect(parakeetModels.map((m) => [m.value, m.localName])).toEqual([
      ["TdtV3Q4K", "parakeet-tdt-0.6b-v3-q4_k"],
      ["TdtV3Q8", "parakeet-tdt-0.6b-v3-q8_0"],
    ]);
    for (const m of [...whisperModels, ...parakeetModels]) {
      expect(modelDownloadSizes[m.localName]).toBe(m.size);
    }
  });

  test("each engine maps to the model it needs", () => {
    expect(engineModelLocalName("WhisperCpp", "Small", "TdtV3Q8")).toBe("whisper-small-q5_1");
    expect(engineModelLocalName("ParakeetCpp", "Small", "TdtV3Q8")).toBe(
      "parakeet-tdt-0.6b-v3-q8_0",
    );
    expect(engineModelLocalName("ParakeetCpp", "Small", "TdtV3Q4K")).toBe(
      "parakeet-tdt-0.6b-v3-q4_k",
    );
    expect(engineModelLocalName("OpenAiCompatible", "Small", "TdtV3Q8")).toBeNull();
  });

  test("Parakeet covers its 25 European languages and nothing else", () => {
    expect(parakeetLanguages.length).toBe(25);
    expect(new Set(parakeetLanguages).size).toBe(25);
    for (const tag of ["auto", "", "en", "EN", "de", "pt-BR", "uk", "es_MX"]) {
      expect(parakeetSupportsLanguage(tag)).toBe(true);
    }
    for (const tag of ["ja", "ko", "zh", "ar", "eng"]) {
      expect(parakeetSupportsLanguage(tag)).toBe(false);
    }
  });

  test("a language list is named, sorted and joined for the locale", () => {
    expect(formatLanguageList("en", ["fr", "de", "en"])).toBe("English, French, and German");
    expect(formatLanguageList("en", parakeetLanguages)).toContain("Ukrainian");
    expect(formatLanguageList("fr", ["de", "en"])).toBe("allemand et anglais");
  });

  test("Parakeet offers its own languages and no Auto-detect", () => {
    const values = dictationLanguagesFor("ParakeetCpp").map((o) => o.value);
    expect(values).not.toContain("auto");
    expect(values.length).toBe(25);
    for (const v of values) expect(parakeetLanguages).toContain(v);
  });

  test("Whisper and the remote API offer every language, Auto first", () => {
    for (const backend of ["WhisperCpp", "OpenAiCompatible"] as const) {
      const values = dictationLanguagesFor(backend).map((o) => o.value);
      expect(values[0]).toBe("auto");
      expect(values.length).toBe(dictationLanguages.length);
      // Every Parakeet language stays selectable after switching back.
      for (const code of parakeetLanguages) expect(values).toContain(code);
    }
  });

  test("a saved language the engine cannot use stays visible, first", () => {
    const options = dictationLanguagesFor("ParakeetCpp", "auto").map((o) => o.value);
    expect(options[0]).toBe("auto");
    expect(options.length).toBe(26);
    expect(dictationLanguagesFor("ParakeetCpp", "de").length).toBe(25);
  });

  test("only Parakeet refuses Auto", () => {
    expect(engineAcceptsLanguage("ParakeetCpp", "auto")).toBe(false);
    expect(engineAcceptsLanguage("ParakeetCpp", "ja")).toBe(false);
    expect(engineAcceptsLanguage("ParakeetCpp", "de")).toBe(true);
    expect(engineAcceptsLanguage("WhisperCpp", "auto")).toBe(true);
    expect(engineAcceptsLanguage("WhisperCpp", "ja")).toBe(true);
  });

  test("switching to Parakeet picks the system language it covers, else English", () => {
    expect(languageForEngine("ParakeetCpp", "auto", ["de-AT", "en-US"])).toBe("de");
    expect(languageForEngine("ParakeetCpp", "ja", ["ja-JP", "fr"])).toBe("fr");
    expect(languageForEngine("ParakeetCpp", "zh", ["zh-CN"])).toBe("en");
    expect(languageForEngine("ParakeetCpp", "auto", [])).toBe("en");
    // A language it already takes is kept.
    expect(languageForEngine("ParakeetCpp", "pl", ["de"])).toBe("pl");
    // Whisper takes anything, so it never moves the language.
    expect(languageForEngine("WhisperCpp", "auto", ["de"])).toBe("auto");
    expect(languageForEngine("WhisperCpp", "pl", [])).toBe("pl");
  });

  test("the dictation languages Parakeet cannot do are exactly the East Asian ones", () => {
    const unsupported = dictationLanguages
      .map((o) => o.value)
      .filter((v) => !parakeetSupportsLanguage(v));
    expect(unsupported.sort()).toEqual(["ja", "ko", "zh"]);
  });

  test("deliveryMethods has both send methods", () => {
    const values = deliveryMethods.map((o) => o.value);
    expect(values).toContain("Clipboard");
    expect(values).toContain("TypeInPlace");
    expect(values).not.toContain("Both");
  });

  test("pillVisibilityOptions has all three modes", () => {
    const values = pillVisibilityOptions.map((o) => o.value);
    expect(values).toContain("Always");
    expect(values).toContain("WhenDictating");
    expect(values).toContain("Never");
  });

  test("history retention offers off, windows, and keep-forever", () => {
    const values = historyRetentionOptions.map((o) => o.value);
    expect(values).toContain("off");
    expect(values).toContain("30");
    expect(values).toContain("forever");
  });

  // Three choices ride in one nullable number, and two of them are not a
  // window: null is off, 0 is keep-forever. Confusing 0 with a zero-day window
  // would prune every session on the next save.
  test("history retention maps the stored number both ways", () => {
    expect(historyRetentionValue(null)).toBe("off");
    expect(historyRetentionValue(0)).toBe("forever");
    expect(historyRetentionValue(30)).toBe("30");

    expect(historyRetentionDays("off")).toBeNull();
    expect(historyRetentionDays("forever")).toBe(0);
    expect(historyRetentionDays("90")).toBe(90);
  });

  test("every history retention option round-trips through the stored value", () => {
    for (const option of historyRetentionOptions) {
      expect(historyRetentionValue(historyRetentionDays(option.value))).toBe(option.value);
    }
  });

  test("labelFor returns the label for a known value", () => {
    expect(labelFor(i18n, activationModes, "Toggle")).toContain("Toggle");
    expect(labelFor(i18n, asrBackends, "WhisperCpp")).toContain("Whisper");
    expect(labelFor(i18n, deliveryMethods, "Clipboard")).toContain("Paste");
  });

  test("labelFor returns the value itself for unknown values", () => {
    expect(labelFor(i18n, activationModes, "UnknownMode" as any)).toBe("UnknownMode");
  });

  test("all options have non-empty labels", () => {
    const allOptions = [
      ...activationModes,
      ...asrBackends,
      ...deliveryMethods,
      ...pillVisibilityOptions,
    ];
    for (const opt of allOptions) {
      expect(localizeText(i18n, opt.label).length).toBeGreaterThan(0);
    }
  });

  test("no duplicate values within each option set", () => {
    for (const options of [activationModes, asrBackends, deliveryMethods, pillVisibilityOptions]) {
      const values = options.map((o) => o.value);
      expect(new Set(values).size).toBe(values.length);
    }
  });
});
