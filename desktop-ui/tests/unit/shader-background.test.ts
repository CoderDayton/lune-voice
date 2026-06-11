import { describe, expect, test } from "bun:test";
import { parseShaderCssColor } from "../../src/features/shell/components/shader-background";

describe("shader background color parsing", () => {
  test("parses rgb theme colors", () => {
    expect(parseShaderCssColor("rgb(255 128 0)")).toEqual([1, 128 / 255, 0]);
    expect(parseShaderCssColor("rgba(0, 128, 255, 0.5)")).toEqual([0, 128 / 255, 1]);
  });

  test("parses oklch theme colors into normalized srgb channels", () => {
    const color = parseShaderCssColor("oklch(0.74 0.14 295)");
    expect(color).not.toBeNull();
    for (const channel of color!) {
      expect(channel).toBeGreaterThanOrEqual(0);
      expect(channel).toBeLessThanOrEqual(1);
    }
  });

  test("returns null for unsupported color formats", () => {
    expect(parseShaderCssColor("var(--color-primary)")).toBeNull();
  });
});
