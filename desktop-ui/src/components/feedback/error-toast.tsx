import { useStore } from "@tanstack/react-store";
import { CircleX, X } from "lucide-react";
import { workspaceActions, workspaceStore } from "../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function ErrorToast() {
  const { t } = useLingui();
  const error = useStore(workspaceStore, (state) => state.runtime.error);

  if (!error) return null;

  return (
    <div className="toast toast-end toast-bottom z-50 pointer-events-none">
      <div role="alert" aria-live="assertive" className="alert alert-error alert-soft pointer-events-auto max-w-sm shadow-lg">
        <CircleX size={18} strokeWidth={2} className="shrink-0" />
        <span className="text-sm">{error}</span>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-circle"
          aria-label={t`Dismiss error`}
          onClick={() => workspaceActions.setError(null)}
        >
          <X size={14} strokeWidth={2} />
        </button>
      </div>
    </div>
  );
}
