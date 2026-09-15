// SPDX-License-Identifier: GPL-3.0-or-later

import type { MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';

/** Build a combo string from a KeyboardEvent (e.g. "Ctrl+Shift+F5"). */
export function buildCombo(event: KeyboardEvent): string | null {
  const modifierKeys = new Set(['Control', 'Shift', 'Alt', 'Meta']);
  if (modifierKeys.has(event.key)) {
    return null;
  }

  const key = normalizeKey(event.key);
  if (!key) return null;

  const parts: string[] = [];
  if (event.ctrlKey) parts.push('Ctrl');
  if (event.shiftKey) parts.push('Shift');
  if (event.altKey) parts.push('Alt');
  if (event.metaKey) parts.push('Meta');
  parts.push(key);

  return parts.join('+');
}

function normalizeKey(key: string): string {
  if (key === ' ') return 'Space';
  // Map browser KeyboardEvent.key values to the single-token names the
  // Rust backend expects (case-insensitive match in key_name_to_code).
  const MAP: Record<string, string> = {
    // Navigation
    ArrowUp: 'Up',
    ArrowDown: 'Down',
    ArrowLeft: 'Left',
    ArrowRight: 'Right',
    ContextMenu: 'Menu',
    // Punctuation — browsers return the character, backend expects names
    '-': 'Minus',
    '=': 'Equal',
    '[': 'BracketLeft',
    ']': 'BracketRight',
    '\\': 'Backslash',
    ';': 'Semicolon',
    "'": 'Quote',
    '`': 'Grave',
    ',': 'Comma',
    '.': 'Period',
    '/': 'Slash',
  };
  return MAP[key] ?? key;
}

// Keys the Rust evdev hotkey listener can resolve (key_name_to_code).
// Must stay in sync with crates/lune-voice/src/hotkey_listener.rs.
const SUPPORTED_KEYS = new Set([
  'space',
  'escape',
  'esc',
  'return',
  'enter',
  'tab',
  'backspace',
  'delete',
  'insert',
  'home',
  'end',
  'pageup',
  'pagedown',
  'up',
  'down',
  'left',
  'right',
  'capslock',
  'numlock',
  'scrolllock',
  'pause',
  'printscreen',
  'sysrq',
  'menu',
  'compose',
  'f1',
  'f2',
  'f3',
  'f4',
  'f5',
  'f6',
  'f7',
  'f8',
  'f9',
  'f10',
  'f11',
  'f12',
  'a',
  'b',
  'c',
  'd',
  'e',
  'f',
  'g',
  'h',
  'i',
  'j',
  'k',
  'l',
  'm',
  'n',
  'o',
  'p',
  'q',
  'r',
  's',
  't',
  'u',
  'v',
  'w',
  'x',
  'y',
  'z',
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  'minus',
  'equal',
  'equals',
  'leftbrace',
  'bracketleft',
  'rightbrace',
  'bracketright',
  'backslash',
  'semicolon',
  'apostrophe',
  'quote',
  'grave',
  'backtick',
  'comma',
  'dot',
  'period',
  'slash',
]);

/**
 * Validate combo: non-modifier required, single letters need modifier, key must
 * be backend-supported. Returns the failure as a message to translate at the
 * point it is shown, so the rule stays usable outside React.
 */
export function validateCombo(combo: string): MessageDescriptor | null {
  const parts = combo.split('+');
  const modifierSet = new Set(['Ctrl', 'Shift', 'Alt', 'Meta']);
  const nonModifiers = parts.filter((p) => !modifierSet.has(p));

  if (nonModifiers.length === 0) {
    return msg`Combo must include a non-modifier key.`;
  }

  const triggerKey = nonModifiers[nonModifiers.length - 1];
  const hasModifier = parts.length > nonModifiers.length;

  if (triggerKey.length === 1 && !hasModifier) {
    return msg`Single letter keys require at least one modifier (Ctrl, Shift, Alt, or Meta).`;
  }

  if (!SUPPORTED_KEYS.has(triggerKey.toLowerCase())) {
    return msg`"${triggerKey}" isn't supported as a hotkey.`;
  }

  return null;
}
