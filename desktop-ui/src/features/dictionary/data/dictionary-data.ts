export interface DictionaryEntry {
  id: number;
  phrase: string;
  replacement: string;
  priority: number;
}

export const initialDictionaryEntries: DictionaryEntry[] = [
  { id: 1, phrase: "github", replacement: "GitHub", priority: 5 },
  { id: 2, phrase: "kubernetes", replacement: "Kubernetes", priority: 8 },
  { id: 3, phrase: "lune voice", replacement: "Lune Voice", priority: 10 },
  { id: 4, phrase: "rust lang", replacement: "Rust", priority: 5 },
  { id: 5, phrase: "api", replacement: "API", priority: 6 },
  { id: 6, phrase: "llm", replacement: "LLM", priority: 7 },
];
