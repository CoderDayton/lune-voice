#!/usr/bin/env bun
// SPDX-License-Identifier: GPL-3.0-or-later
/**
 * Prints the licence notices for the JavaScript that ships inside the app's
 * webview bundle (`dist/`), as Markdown.
 *
 * Walks `dependencies` from package.json through node_modules, following each
 * package's own `dependencies` and `optionalDependencies`. Peer dependencies are
 * the host's to provide, so they are counted where the host declares them. The
 * build-time packages whose output is copied into the bundle (see
 * BUNDLED_BUILD_DEPS) are listed without their own dependency trees, which are
 * build tooling. Called by `scripts/release/third-party-licenses.sh`.
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

/**
 * devDependencies whose code or CSS lands in `dist/`: Tailwind's preflight and
 * utilities, daisyUI's component styles, and Vite's module preload helper.
 */
const BUNDLED_BUILD_DEPS = ['tailwindcss', 'daisyui', 'vite'];

export interface PackageNotice {
  name: string;
  version: string;
  license: string;
  repository: string | null;
  /** Licence and NOTICE files, joined; `null` when the package ships none. */
  text: string | null;
}

interface Manifest {
  name?: string;
  version?: string;
  license?: string | { type?: string };
  repository?: string | { url?: string };
  dependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
}

/** `LICENSE`, `LICENSE.md`, `LICENSE-MIT`, `LICENSE_APACHE-2.0`, `COPYING`, `NOTICE`… */
const LICENSE_FILE = /^(licen[cs]e|copying|notice)([._-].*)?$/i;

function readManifest(dir: string): Manifest {
  return JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')) as Manifest;
}

/** Node's lookup: the nearest `node_modules/<name>` from `from` up to `root`. */
function resolve(from: string, name: string, root: string): string | null {
  let dir = from;
  for (;;) {
    const candidate = join(dir, 'node_modules', name);
    if (existsSync(join(candidate, 'package.json'))) return candidate;
    if (dir === root) return null;
    const parent = dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

function licenseText(dir: string): string | null {
  const files = readdirSync(dir)
    .filter((f) => LICENSE_FILE.test(f))
    .sort();
  if (files.length === 0) return null;
  return files.map((f) => readFileSync(join(dir, f), 'utf8').trim()).join('\n\n');
}

function repositoryUrl(repo: Manifest['repository']): string | null {
  const raw = typeof repo === 'string' ? repo : repo?.url;
  if (!raw) return null;
  return raw.replace(/^git\+/, '').replace(/\.git$/, '');
}

export function collectNotices(root: string, extra: string[] = BUNDLED_BUILD_DEPS): PackageNotice[] {
  const top = readManifest(root);
  const seen = new Map<string, PackageNotice>();
  const queue: Array<[from: string, name: string, walk: boolean]> = [
    ...Object.keys(top.dependencies ?? {}).map((name): [string, string, boolean] => [root, name, true]),
    ...extra.map((name): [string, string, boolean] => [root, name, false]),
  ];

  while (queue.length > 0) {
    const [from, name, walk] = queue.shift() as [string, string, boolean];
    const dir = resolve(from, name, root);
    // An optional dependency that is not installed is not shipped.
    if (!dir) continue;
    const manifest = readManifest(dir);
    const key = `${manifest.name ?? name}@${manifest.version ?? ''}`;
    if (seen.has(key)) continue;

    const license =
      typeof manifest.license === 'string' ? manifest.license : (manifest.license?.type ?? 'UNKNOWN');
    seen.set(key, {
      name: manifest.name ?? name,
      version: manifest.version ?? '',
      license,
      repository: repositoryUrl(manifest.repository),
      text: licenseText(dir),
    });

    if (!walk) continue;
    for (const dep of [
      ...Object.keys(manifest.dependencies ?? {}),
      ...Object.keys(manifest.optionalDependencies ?? {}),
    ]) {
      queue.push([dir, dep, true]);
    }
  }

  return [...seen.values()].sort((a, b) =>
    a.name === b.name ? a.version.localeCompare(b.version) : a.name.localeCompare(b.name),
  );
}

export function renderNotices(notices: PackageNotice[]): string {
  const sections = notices.map((n) => {
    const lines = [`### ${n.name} ${n.version}`, '', `License: ${n.license}`];
    if (n.repository) lines.push(`Source: ${n.repository}`);
    lines.push('');
    if (n.text) {
      lines.push('````text', n.text, '````');
    } else {
      lines.push(`This package ships no licence file; it declares \`${n.license}\`.`);
    }
    return lines.join('\n');
  });
  return [
    '## JavaScript packages in the interface bundle',
    '',
    ...sections.flatMap((s) => [s, '']),
  ].join('\n');
}

if (import.meta.main) {
  process.stdout.write(renderNotices(collectNotices(join(import.meta.dir, '..'))));
}
