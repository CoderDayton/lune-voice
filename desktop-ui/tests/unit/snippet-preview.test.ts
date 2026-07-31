import { describe, expect, test } from 'bun:test';
import { previewBody } from '../../src/features/snippets/snippet-preview';

const CLIP = '(clipboard)';

describe('previewBody', () => {
  test('substitutes known placeholders regardless of case', () => {
    expect(previewBody('{clipboard}', CLIP)).toBe(CLIP);
    expect(previewBody('{CLIPBOARD}', CLIP)).toBe(CLIP);
    expect(previewBody('a {clipboard} b', CLIP)).toBe(`a ${CLIP} b`);
  });

  test('leaves unknown tokens and their braces alone', () => {
    expect(previewBody('{nope}', CLIP)).toBe('{nope}');
    expect(previewBody('{}', CLIP)).toBe('{}');
  });

  test('leaves an unclosed brace and everything after it verbatim', () => {
    // `render_body` bails out at the first `{` with no `}` after it.
    expect(previewBody('a {clipboard', CLIP)).toBe('a {clipboard');
    expect(previewBody('{clipboard} then {oops', CLIP)).toBe(`${CLIP} then {oops`);
  });

  test('matches the backend on a nested opening brace', () => {
    // The token is "a{clipboard", which resolves to nothing, so the whole run
    // survives. A `\{([^{}]*)\}` regex would find the inner "{clipboard}" and
    // substitute — a preview the expansion never produces.
    expect(previewBody('{a{clipboard}', CLIP)).toBe('{a{clipboard}');
  });

  test('passes text with no placeholders straight through', () => {
    expect(previewBody('plain text', CLIP)).toBe('plain text');
    expect(previewBody('', CLIP)).toBe('');
  });
});
