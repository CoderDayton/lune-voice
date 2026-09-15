// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from "bun:test";
import {
  type DictionaryEntry,
  entryMatchesQuery,
  VOCAB_PRESETS,
} from "../../src/features/dictionary/data/dictionary-data";

const entry: DictionaryEntry = {
  id: 1,
  phrase: "github",
  replacement: "GitHub",
  priority: 5,
};

describe("entryMatchesQuery", () => {
  test("empty / whitespace query matches everything", () => {
    expect(entryMatchesQuery(entry, "")).toBe(true);
    expect(entryMatchesQuery(entry, "   ")).toBe(true);
  });

  test("matches on the phrase, case-insensitively", () => {
    expect(entryMatchesQuery(entry, "git")).toBe(true);
    expect(entryMatchesQuery(entry, "GIT")).toBe(true);
  });

  test("matches on the replacement", () => {
    expect(entryMatchesQuery(entry, "hub")).toBe(true);
  });

  test("trims surrounding whitespace before matching", () => {
    expect(entryMatchesQuery(entry, "  github  ")).toBe(true);
  });

  test("returns false when neither field contains the query", () => {
    expect(entryMatchesQuery(entry, "kubernetes")).toBe(false);
  });
});

describe("VOCAB_PRESETS", () => {
  test("every preset is a complete, valid rule", () => {
    expect(VOCAB_PRESETS.length).toBeGreaterThan(0);
    for (const preset of VOCAB_PRESETS) {
      expect(preset.phrase.trim().length).toBeGreaterThan(0);
      expect(preset.replacement.trim().length).toBeGreaterThan(0);
      expect(preset.priority).toBeGreaterThanOrEqual(1);
      expect(preset.priority).toBeLessThanOrEqual(9);
    }
  });

  test("preset phrases are unique", () => {
    const phrases = VOCAB_PRESETS.map((p) => p.phrase);
    expect(new Set(phrases).size).toBe(phrases.length);
  });
});
