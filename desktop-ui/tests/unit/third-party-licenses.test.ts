// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { collectNotices, renderNotices } from '../../scripts/third-party-licenses';

let root: string;

function pkg(dir: string, manifest: Record<string, unknown>, license?: string) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify(manifest));
  if (license) writeFileSync(join(dir, 'LICENSE'), license);
}

afterEach(() => rmSync(root, { recursive: true, force: true }));

describe('third-party licence collection', () => {
  test('walks runtime dependencies, nested installs and named build deps only', () => {
    root = mkdtempSync(join(tmpdir(), 'notices-'));
    pkg(root, {
      name: 'app',
      dependencies: { a: '^1' },
      devDependencies: { tooling: '^1', 'css-lib': '^1' },
    });
    pkg(
      join(root, 'node_modules/a'),
      {
        name: 'a',
        version: '1.0.0',
        license: 'MIT',
        dependencies: { d: '^2' },
        optionalDependencies: { 'not-installed': '^1' },
      },
      'MIT License\nCopyright (c) A',
    );
    pkg(
      join(root, 'node_modules/a/node_modules/d'),
      { name: 'd', version: '2.0.0', license: { type: 'ISC' } },
      'ISC text',
    );
    pkg(join(root, 'node_modules/tooling'), { name: 'tooling', version: '1.0.0', license: 'MIT' });
    pkg(
      join(root, 'node_modules/css-lib'),
      { name: 'css-lib', version: '3.0.0', license: 'MIT' },
      'css text',
    );

    const notices = collectNotices(root, ['css-lib']);

    expect(notices.map((n) => `${n.name}@${n.version}`)).toEqual([
      'a@1.0.0',
      'css-lib@3.0.0',
      'd@2.0.0',
    ]);
    expect(notices.find((n) => n.name === 'd')?.license).toBe('ISC');
    expect(notices.find((n) => n.name === 'a')?.text).toContain('Copyright (c) A');
  });

  test('peers are the host’s to ship, and a build dep’s own tooling is not bundled', () => {
    root = mkdtempSync(join(tmpdir(), 'notices-'));
    pkg(root, { name: 'app', dependencies: { ui: '^1' }, devDependencies: { bundler: '^1' } });
    pkg(join(root, 'node_modules/ui'), {
      name: 'ui',
      version: '1.0.0',
      license: 'MIT',
      peerDependencies: { 'babel-macro': '^1' },
    });
    pkg(join(root, 'node_modules/babel-macro'), { name: 'babel-macro', version: '1.0.0' });
    pkg(join(root, 'node_modules/bundler'), {
      name: 'bundler',
      version: '8.0.0',
      license: 'MIT',
      dependencies: { compiler: '^1' },
    });
    pkg(join(root, 'node_modules/compiler'), { name: 'compiler', version: '1.0.0' });

    const names = collectNotices(root, ['bundler']).map((n) => n.name);

    expect(names).toEqual(['bundler', 'ui']);
  });

  test('licence files named per licence are read', () => {
    root = mkdtempSync(join(tmpdir(), 'notices-'));
    pkg(root, { name: 'app', dependencies: { dual: '^1' } });
    pkg(join(root, 'node_modules/dual'), { name: 'dual', version: '2.0.0', license: 'MIT OR Apache-2.0' });
    writeFileSync(join(root, 'node_modules/dual/LICENSE_MIT'), 'mit text');
    writeFileSync(join(root, 'node_modules/dual/LICENSE_APACHE-2.0'), 'apache text');

    const [dual] = collectNotices(root, []);

    expect(dual?.text).toContain('mit text');
    expect(dual?.text).toContain('apache text');
  });

  test('a package without a licence file is still listed by its declared licence', () => {
    root = mkdtempSync(join(tmpdir(), 'notices-'));
    pkg(root, { name: 'app', dependencies: { bare: '^1' } });
    pkg(join(root, 'node_modules/bare'), { name: 'bare', version: '0.1.0', license: 'MIT' });

    const out = renderNotices(collectNotices(root, []));

    expect(out).toContain('bare 0.1.0');
    expect(out).toContain('MIT');
  });
});
