import { describe, expect, test } from "bun:test";
import { buildCombo, validateCombo } from "../../src/features/settings/components/hotkey-utils";

function makeKeyEvent(overrides: Partial<KeyboardEvent> = {}): KeyboardEvent {
  return {
    key: "a",
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    metaKey: false,
    ...overrides,
  } as KeyboardEvent;
}

describe("buildCombo", () => {
  test("returns null for bare modifier keys", () => {
    expect(buildCombo(makeKeyEvent({ key: "Control" }))).toBeNull();
    expect(buildCombo(makeKeyEvent({ key: "Shift" }))).toBeNull();
    expect(buildCombo(makeKeyEvent({ key: "Alt" }))).toBeNull();
    expect(buildCombo(makeKeyEvent({ key: "Meta" }))).toBeNull();
  });

  test("builds simple key combo", () => {
    expect(buildCombo(makeKeyEvent({ key: "a" }))).toBe("a");
  });

  test("builds combo with single modifier", () => {
    expect(buildCombo(makeKeyEvent({ key: "a", ctrlKey: true }))).toBe("Ctrl+a");
  });

  test("builds combo with multiple modifiers", () => {
    const result = buildCombo(makeKeyEvent({ key: "a", ctrlKey: true, shiftKey: true }));
    expect(result).toBe("Ctrl+Shift+a");
  });

  test("builds combo with all modifiers", () => {
    const result = buildCombo(
      makeKeyEvent({ key: "F5", ctrlKey: true, shiftKey: true, altKey: true, metaKey: true }),
    );
    expect(result).toBe("Ctrl+Shift+Alt+Meta+F5");
  });

  test("normalizes space key", () => {
    expect(buildCombo(makeKeyEvent({ key: " ", ctrlKey: true }))).toBe("Ctrl+Space");
  });

  test("normalizes arrow keys", () => {
    expect(buildCombo(makeKeyEvent({ key: "ArrowUp" }))).toBe("Up");
    expect(buildCombo(makeKeyEvent({ key: "ArrowDown" }))).toBe("Down");
    expect(buildCombo(makeKeyEvent({ key: "ArrowLeft" }))).toBe("Left");
    expect(buildCombo(makeKeyEvent({ key: "ArrowRight" }))).toBe("Right");
  });

  test("normalizes punctuation to backend-compatible names", () => {
    expect(buildCombo(makeKeyEvent({ key: "-" }))).toBe("Minus");
    expect(buildCombo(makeKeyEvent({ key: "=" }))).toBe("Equal");
    expect(buildCombo(makeKeyEvent({ key: ";" }))).toBe("Semicolon");
    expect(buildCombo(makeKeyEvent({ key: "'" }))).toBe("Quote");
    expect(buildCombo(makeKeyEvent({ key: "`" }))).toBe("Grave");
    expect(buildCombo(makeKeyEvent({ key: "," }))).toBe("Comma");
    expect(buildCombo(makeKeyEvent({ key: "." }))).toBe("Period");
    expect(buildCombo(makeKeyEvent({ key: "/" }))).toBe("Slash");
    expect(buildCombo(makeKeyEvent({ key: "\\" }))).toBe("Backslash");
  });

  test("preserves function keys as-is", () => {
    expect(buildCombo(makeKeyEvent({ key: "F1" }))).toBe("F1");
    expect(buildCombo(makeKeyEvent({ key: "F12" }))).toBe("F12");
  });

  test("preserves named keys like Escape, Enter", () => {
    expect(buildCombo(makeKeyEvent({ key: "Escape" }))).toBe("Escape");
    expect(buildCombo(makeKeyEvent({ key: "Enter" }))).toBe("Enter");
    expect(buildCombo(makeKeyEvent({ key: "Tab" }))).toBe("Tab");
  });
});

describe("validateCombo", () => {
  test("accepts valid combos with modifier + letter", () => {
    expect(validateCombo("Ctrl+Space")).toBeNull();
    expect(validateCombo("Ctrl+Shift+F5")).toBeNull();
    expect(validateCombo("Alt+a")).toBeNull();
    expect(validateCombo("Meta+z")).toBeNull();
  });

  test("accepts function keys without modifier", () => {
    expect(validateCombo("F1")).toBeNull();
    expect(validateCombo("F12")).toBeNull();
  });

  test("accepts special keys without modifier", () => {
    expect(validateCombo("Escape")).toBeNull();
    expect(validateCombo("Space")).toBeNull();
    expect(validateCombo("Pause")).toBeNull();
  });

  test("rejects single letter without modifier", () => {
    const error = validateCombo("a");
    expect(error).toContain("modifier");
  });

  test("rejects unsupported keys", () => {
    const error = validateCombo("Ctrl+AudioVolumeMute");
    expect(error).toContain("isn't supported");
  });

  test("rejects modifier-only combos", () => {
    // This shouldn't normally happen since buildCombo returns null for modifier-only,
    // but validateCombo should catch it defensively
    // Actually "Ctrl" alone wouldn't be passed since it's all modifiers with no trigger
    // Let's test the edge case where parts = ["Ctrl"] — the key "Ctrl" is in modifierSet
    const error = validateCombo("Ctrl");
    expect(error).toContain("non-modifier key");
  });

  test("accepts number keys with modifier", () => {
    expect(validateCombo("Ctrl+1")).toBeNull();
    expect(validateCombo("Alt+0")).toBeNull();
  });

  test("accepts punctuation names", () => {
    expect(validateCombo("Ctrl+Minus")).toBeNull();
    expect(validateCombo("Ctrl+Comma")).toBeNull();
    expect(validateCombo("Shift+Grave")).toBeNull();
  });

  test("validation is case-insensitive for the key", () => {
    expect(validateCombo("Ctrl+SPACE")).toBeNull();
    expect(validateCombo("Ctrl+escape")).toBeNull();
  });
});
