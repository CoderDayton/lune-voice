export type ActivationModeValue = "Toggle" | "Hold";
export type AsrBackendValue = "OrtQwenAsr" | "ParakeetTdt";
export type LlmBackendValue = "LlamaCppLocal" | "OpenAiCompatible";
export type DictationModeValue = "TypeInPlace" | "Clipboard" | "Both";

export interface Option<Value extends string> {
  value: Value;
  label: string;
}

export const activationModes: Option<ActivationModeValue>[] = [
  { value: "Toggle", label: "Toggle to start and stop" },
  { value: "Hold", label: "Hold while speaking" },
];

export const asrBackends: Option<AsrBackendValue>[] = [
  { value: "OrtQwenAsr", label: "Balanced (recommended)" },
  { value: "ParakeetTdt", label: "Fast and light" },
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

function labelForOption<Value extends string>(options: Option<Value>[], value: Value) {
  return options.find((option) => option.value === value)?.label ?? value;
}

export function labelForActivationMode(value: ActivationModeValue) {
  return labelForOption(activationModes, value);
}

export function labelForAsrBackend(value: AsrBackendValue) {
  return labelForOption(asrBackends, value);
}

export function labelForLlmBackend(value: LlmBackendValue) {
  return labelForOption(llmModels, value);
}

export function labelForDeliveryMethod(value: DictationModeValue) {
  return labelForOption(deliveryMethods, value);
}

export function labelForStylePreset(value: string) {
  return labelForOption(stylePresets, value);
}
