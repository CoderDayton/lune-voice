import { useStore } from "@tanstack/react-store";
import { workspaceStore, saveWorkspaceSetting } from "../../store/workspace-store";
import { stylePresets } from "../settings/data/settings-data";
import { Paintbrush, Check } from "lucide-react";
import { useLingui } from "@lingui/react/macro";

export function StylesPage() {
  const { t } = useLingui();
  const activePreset = useStore(workspaceStore, (s) => s.settings.stylePreset);

  return (
    <div className="page-stack">
      <div className="flex items-center gap-3 mb-2">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/8">
          <Paintbrush size={18} strokeWidth={1.8} className="text-primary/60" />
        </div>
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{t`Writing styles`}</h2>
          <p className="text-sm text-content-subtle mt-0.5">
            {t`Choose the tone and formatting applied to your transcriptions.`}
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {stylePresets.map((preset) => {
          const isActive = activePreset === preset.value;
          return (
            <button
              key={preset.value}
              type="button"
              className={`card border rounded-box p-5 text-left transition-colors cursor-pointer ${
                isActive
                  ? "border-primary bg-primary/5"
                  : "border-surface-border bg-surface-card hover:border-primary/40"
              }`}
              onClick={() => void saveWorkspaceSetting("stylePreset", preset.value)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">{preset.label}</span>
                {isActive && <Check size={16} className="text-primary" />}
              </div>
              <p className="text-xs text-content-subtle">
                {styleDescription(preset.value)}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function styleDescription(value: string): string {
  switch (value) {
    case "casual":
      return "Relaxed, conversational tone. Contractions and everyday language.";
    case "professional":
      return "Polished and clear. Suitable for emails, documents, and meetings.";
    case "technical":
      return "Precise formatting for code discussions, technical docs, and engineering notes.";
    default:
      return "";
  }
}
