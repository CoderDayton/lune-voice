// SPDX-License-Identifier: GPL-3.0-or-later

import { setupI18n } from "@lingui/core";
import { describe, expect, test } from "bun:test";
import { formatDuration, sessionStatusLabel, transcriptFileName } from "../../src/lib/session-format";

describe("sessionStatusLabel", () => {
  const i18n = setupI18n({ locale: "en", messages: { en: {} } });
  i18n.activate("en");

  test("known states come from the catalog and a failure keeps its reason", () => {
    expect(sessionStatusLabel(i18n, "Completed")).toBe("Completed");
    expect(sessionStatusLabel(i18n, "Cancelled")).toBe("Cancelled");
    expect(sessionStatusLabel(i18n, "Active")).toBe("In progress");
    expect(sessionStatusLabel(i18n, "Failed: microphone is in use")).toBe(
      "Failed: microphone is in use",
    );
    expect(sessionStatusLabel(i18n, "Something new")).toBe("Something new");
  });
});

describe("formatDuration", () => {
  test("renders m:ss with a zero-padded seconds field", () => {
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(5)).toBe("0:05");
    expect(formatDuration(65)).toBe("1:05");
    expect(formatDuration(125)).toBe("2:05");
  });

  test("keeps whole minutes past an hour rather than rolling over", () => {
    expect(formatDuration(3661)).toBe("61:01");
  });
});

describe("transcriptFileName", () => {
  test("stamps the local start time down to the minute", () => {
    const local = new Date(2026, 2, 7, 9, 5).toISOString();
    expect(transcriptFileName(local)).toBe("lune-2026-03-07-0905.txt");
  });
});
