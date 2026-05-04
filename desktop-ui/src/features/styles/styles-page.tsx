import { useStore } from "@tanstack/react-store";
import { workspaceStore, saveWorkspaceSetting } from "../../store/workspace-store";
import { Paintbrush, Check } from "lucide-react";
import { useLingui } from "@lingui/react/macro";

export function StylesPage() {
  const { t } = useLingui();
  const activePreset = useStore(workspaceStore, (s) => s.settings.stylePreset);
  const presets = useStore(workspaceStore, (s) => s.stylePresets);
  const loading = useStore(workspaceStore, (s) => s.runtime.loading);

  return (
    <div className="page-stack">
      <div className="flex items-center gap-4 mb-2">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/8">
          <Paintbrush size={18} strokeWidth={1.8} className="text-primary/60" />
        </div>
        <div>
          <h2 className="text-xl font-semibold tracking-tight">{t`Writing styles`}</h2>
          <p className="text-sm text-content-subtle mt-0.5">
            {t`Choose the tone and formatting applied to your transcriptions.`}
          </p>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {loading && presets.length === 0
          ? [0, 1, 2].map((i) => (
              <div
                key={i}
                className="card border rounded-box p-5 border-surface-border bg-surface-card"
              >
                <div className="skeleton h-5 w-24 mb-2" />
                <div className="skeleton h-3 w-full" />
              </div>
            ))
          : presets.map((preset) => {
              const isActive = activePreset === preset.id;
              return (
                <button
                  key={preset.id}
                  type="button"
                  className={`card border rounded-box p-5 text-left transition-colors cursor-pointer ${
                    isActive
                      ? "border-primary bg-primary/5"
                      : "border-surface-border bg-surface-card hover:border-primary/40"
                  }`}
                  onClick={() => void saveWorkspaceSetting("stylePreset", preset.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{preset.name}</span>
                    {isActive && <Check size={16} className="text-primary" />}
                  </div>
                  <p className="text-sm text-content-subtle">{preset.description}</p>
                </button>
              );
            })}
      </div>
    </div>
  );
}
