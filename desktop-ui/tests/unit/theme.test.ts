// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from "bun:test";
import { ALL_THEMES, isDarkTheme, themeLabel, THEME_GROUPS } from "../../src/lib/theme";

describe("theme utilities", () => {
  test("ALL_THEMES includes all groups", () => {
    const expected =
      THEME_GROUPS.custom.length + THEME_GROUPS.light.length + THEME_GROUPS.dark.length;
    expect(ALL_THEMES.length).toBe(expected);
  });

  test("ALL_THEMES has no duplicates", () => {
    const unique = new Set(ALL_THEMES);
    expect(unique.size).toBe(ALL_THEMES.length);
  });

  test("themeLabel capitalizes custom themes", () => {
    expect(themeLabel("lune")).toBe("Lune Dark");
    expect(themeLabel("lune-light")).toBe("Lune Light");
  });

  test("themeLabel capitalizes standard themes", () => {
    expect(themeLabel("dark")).toBe("Dark");
    expect(themeLabel("cupcake")).toBe("Cupcake");
    expect(themeLabel("synthwave")).toBe("Synthwave");
  });

  test("isDarkTheme identifies dark themes correctly", () => {
    expect(isDarkTheme("lune")).toBe(true);
    expect(isDarkTheme("dark")).toBe(true);
    expect(isDarkTheme("dracula")).toBe(true);
    expect(isDarkTheme("forest")).toBe(true);
    expect(isDarkTheme("night")).toBe(true);
  });

  test("isDarkTheme identifies light themes correctly", () => {
    expect(isDarkTheme("lune-light")).toBe(false);
    expect(isDarkTheme("light")).toBe(false);
    expect(isDarkTheme("cupcake")).toBe(false);
    expect(isDarkTheme("nord")).toBe(false);
    expect(isDarkTheme("winter")).toBe(false);
  });

  test("custom themes are at the start of ALL_THEMES", () => {
    expect(ALL_THEMES[0]).toBe("lune");
    expect(ALL_THEMES[1]).toBe("lune-light");
  });

  test("lune is in custom group", () => {
    expect(THEME_GROUPS.custom).toContain("lune");
    expect(THEME_GROUPS.custom).toContain("lune-light");
  });
});
