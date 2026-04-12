export type ActivationModeValue = "Toggle" | "Hold";
export type AsrBackendValue = "ParakeetTdt" | "OpenAiCompatible";
export type LlmBackendValue = "LlamaCppLocal" | "OpenAiCompatible"; // LlamaCppLocal is a no-op passthrough; kept for config compat
export type DictationModeValue = "TypeInPlace" | "Clipboard" | "Both";
export type PillVisibilityValue = "Always" | "WhenDictating" | "Never";

export interface Option<Value extends string> {
  value: Value;
  label: string;
}

export const activationModes: Option<ActivationModeValue>[] = [
  { value: "Toggle", label: "Toggle to start and stop" },
  { value: "Hold", label: "Hold while speaking" },
];

export const pillVisibilityOptions: Option<PillVisibilityValue>[] = [
  { value: "Always", label: "Always" },
  { value: "WhenDictating", label: "When dictating" },
  { value: "Never", label: "Never" },
];

export const asrBackends: Option<AsrBackendValue>[] = [
  { value: "ParakeetTdt", label: "Parakeet TDT v3 (local, multilingual)" },
  { value: "OpenAiCompatible", label: "Cloud API (OpenAI-compatible)" },
];

export const llmModels: Option<LlmBackendValue>[] = [
  { value: "LlamaCppLocal", label: "Smart cleanup (local)" },
  { value: "OpenAiCompatible", label: "Smart cleanup (online)" },
];

export const deliveryMethods: Option<DictationModeValue>[] = [
  { value: "Clipboard", label: "Paste into app" },
  { value: "TypeInPlace", label: "Type keystrokes" },
  { value: "Both", label: "Paste and type" },
];

export const stylePresets: Option<string>[] = [
  { value: "casual", label: "Everyday" },
  { value: "professional", label: "Professional" },
  { value: "technical", label: "Code" },
];


export function labelFor<V extends string>(options: Option<V>[], value: V): string {
  return options.find((o) => o.value === value)?.label ?? value;
}
