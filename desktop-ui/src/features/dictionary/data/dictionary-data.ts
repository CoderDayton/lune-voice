// SPDX-License-Identifier: GPL-3.0-or-later

export interface DictionaryEntry {
  id: number;
  phrase: string;
  replacement: string;
  priority: number;
}

/**
 * Example corrections offered on the empty state. One tap seeds a rule, so a
 * first-time user sees the feature work instead of facing a blank form
 * (progressive disclosure — teach by example, lower the activation cost).
 */
export interface VocabPreset {
  phrase: string;
  replacement: string;
  priority: number;
}

export const VOCAB_PRESETS: readonly VocabPreset[] = [
  { phrase: 'github', replacement: 'GitHub', priority: 5 },
  { phrase: 'vs code', replacement: 'VS Code', priority: 5 },
  { phrase: 'type script', replacement: 'TypeScript', priority: 5 },
  { phrase: 'kubernetes', replacement: 'Kubernetes', priority: 5 },
  { phrase: 'post gres', replacement: 'Postgres', priority: 5 },
  { phrase: 'json', replacement: 'JSON', priority: 1 },
] as const;

/** Case-insensitive match of a query against an entry's phrase or replacement. */
export function entryMatchesQuery(entry: DictionaryEntry, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return entry.phrase.toLowerCase().includes(q) || entry.replacement.toLowerCase().includes(q);
}
