import { Check, type LucideIcon } from "lucide-react";
import { useLingui } from "@lingui/react/macro";
import type { DownloadProgress } from "./model-setup";

interface ModelCardProps {
  name: string;
  localName: string;
  description: string;
  required: boolean;
  icon: LucideIcon;
  ready: boolean;
  selected: boolean;
  active: boolean;
  canToggle: boolean;
  progress: DownloadProgress | null;
  pct: number;
  onToggle: () => void;
}

export function ModelCard({
  name, description, required, icon: Icon, ready, selected,
  active, canToggle, progress, pct, onToggle,
}: ModelCardProps) {
  const { t } = useLingui();

  return (
    <button
      type="button"
      disabled={!canToggle}
      onClick={canToggle ? onToggle : undefined}
      className={`card card-border card-sm w-full text-left transition-colors ${
        ready ? "border-success/30 bg-success/5"
        : selected ? "border-primary/40 bg-primary/5"
        : "bg-base-200 opacity-60"
      }`}
    >
      <div className="card-body flex-row items-center gap-2 sm:gap-3">
        <div className={`flex size-8 sm:size-9 shrink-0 items-center justify-center rounded-field ${
          ready ? "bg-success/15"
          : active ? "bg-primary/15"
          : selected ? "bg-primary/10"
          : "bg-base-300"
        }`}>
          {ready ? (
            <Check size={16} strokeWidth={2.5} className="text-success" />
          ) : active ? (
            <span className="loading loading-spinner loading-xs text-primary" />
          ) : selected ? (
            <Check size={16} strokeWidth={2.5} className="text-primary" />
          ) : (
            <Icon size={16} strokeWidth={1.5} className="text-content-subtle" />
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-medium text-xs sm:text-sm">{name}</p>
            {required && (
              <span className="badge badge-xs badge-outline text-content-dim">{t`Required`}</span>
            )}
          </div>
          {active && progress ? (
            <div className="mt-1">
              <progress className="progress progress-primary h-1.5 w-full" value={progress.fileIndex + 1} max={progress.fileCount} />
              <p className="text-xs text-content-dim mt-0.5 truncate">{progress.fileName}</p>
            </div>
          ) : (
            <p className="text-xs text-content-subtle mt-0.5 hidden sm:block">{description}</p>
          )}
        </div>

        <div className="shrink-0">
          {ready ? (
            <div className="badge badge-soft badge-success badge-sm">{t`Ready`}</div>
          ) : active ? (
            <span className="text-xs tabular-nums text-primary font-medium">{pct}%</span>
          ) : null}
        </div>
      </div>
    </button>
  );
}
