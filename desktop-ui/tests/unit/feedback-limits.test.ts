import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import {
  CATEGORIES,
  charCount,
  COOLDOWN_SECONDS,
  MAX_MESSAGE_CHARS,
  MIN_MESSAGE_CHARS,
} from "../../src/components/feedback/limits";

// `submit_feedback` re-checks every limit the dialog gates on, but nothing in
// the type system binds the two copies. Parse the authority and compare, so a
// change on either side fails here instead of surfacing as a Send button that
// enables a request the command rejects.
const RUST = readFileSync(join(import.meta.dir, "../../src-tauri/src/feedback.rs"), "utf8");

function rustConst(pattern: RegExp, name: string): string {
  const match = RUST.match(pattern);
  if (!match?.[1]) throw new Error(`${name} not found in feedback.rs`);
  return match[1];
}

describe("feedback limits stay in sync with feedback.rs", () => {
  test("MAX_MESSAGE_CHARS", () => {
    const rust = rustConst(/const MAX_MESSAGE_CHARS: usize = (\d+);/, "MAX_MESSAGE_CHARS");
    expect(MAX_MESSAGE_CHARS).toBe(Number(rust));
  });

  test("MIN_MESSAGE_CHARS", () => {
    const rust = rustConst(/const MIN_MESSAGE_CHARS: usize = (\d+);/, "MIN_MESSAGE_CHARS");
    expect(MIN_MESSAGE_CHARS).toBe(Number(rust));
  });

  test("COOLDOWN_SECONDS", () => {
    const rust = rustConst(
      /const COOLDOWN: Duration = Duration::from_secs\((\d+)\);/,
      "COOLDOWN",
    );
    expect(COOLDOWN_SECONDS).toBe(Number(rust));
  });

  test("CATEGORIES match, in order", () => {
    const block = rustConst(/const CATEGORIES: \[&str; \d+\] = \[([\s\S]*?)\];/, "CATEGORIES");
    const rust = [...block.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
    expect(rust).toEqual([...CATEGORIES]);
  });

  test("the declared CATEGORIES length matches the list", () => {
    const declared = rustConst(/const CATEGORIES: \[&str; (\d+)\]/, "CATEGORIES length");
    expect(CATEGORIES.length).toBe(Number(declared));
  });
});

describe("charCount", () => {
  test("counts code points, not UTF-16 units", () => {
    // The gate the backend applies is `chars().count()`. Five emoji are 10
    // UTF-16 units, which would pass a `.length >= 10` check the command then
    // rejects for being too short.
    expect(charCount("🙂🙂🙂🙂🙂")).toBe(5);
    expect("🙂🙂🙂🙂🙂".length).toBe(10);
  });

  test("matches .length for plain ASCII", () => {
    expect(charCount("it broke")).toBe(8);
  });
});
