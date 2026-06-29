import { describe, expect, test } from "bun:test";
import {
  activationModes,
  asrBackends,
  deliveryMethods,
  labelFor,
  pillVisibilityOptions,
} from "../../src/features/settings/data/settings-data";

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

  test("labelFor returns the label for a known value", () => {
    expect(labelFor(activationModes, "Toggle")).toContain("Toggle");
    expect(labelFor(asrBackends, "WhisperCpp")).toContain("Whisper");
    expect(labelFor(deliveryMethods, "Clipboard")).toContain("Paste");
  });

  test("labelFor returns the value itself for unknown values", () => {
    expect(labelFor(activationModes, "UnknownMode" as any)).toBe("UnknownMode");
  });

  test("all options have non-empty labels", () => {
    const allOptions = [
      ...activationModes,
      ...asrBackends,
      ...deliveryMethods,
      ...pillVisibilityOptions,
    ];
    for (const opt of allOptions) {
      expect(opt.label.length).toBeGreaterThan(0);
    }
  });

  test("no duplicate values within each option set", () => {
    for (const options of [activationModes, asrBackends, deliveryMethods, pillVisibilityOptions]) {
      const values = options.map((o) => o.value);
      expect(new Set(values).size).toBe(values.length);
    }
  });
});
