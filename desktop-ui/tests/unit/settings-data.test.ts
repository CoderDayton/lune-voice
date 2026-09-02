import { setupI18n } from "@lingui/core";
import { describe, expect, test } from "bun:test";
import {
  activationModes,
  asrBackends,
  deliveryMethods,
  historyRetentionDays,
  historyRetentionOptions,
  historyRetentionValue,
  labelFor,
  localizeText,
  pillVisibilityOptions,
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

  test("asrBackends has WhisperCpp and OpenAiCompatible", () => {
    const values = asrBackends.map((o) => o.value);
    expect(values).toContain("WhisperCpp");
    expect(values).toContain("OpenAiCompatible");
    expect(values).not.toContain("Qwen3Asr");
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
