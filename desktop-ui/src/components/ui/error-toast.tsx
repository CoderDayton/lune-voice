import { useStore } from "@tanstack/react-store";
import { workspaceActions, workspaceStore } from "../../store/workspace-store";
import { useLingui } from "@lingui/react/macro";

export function ErrorToast() {
  const { t } = useLingui();
  const error = useStore(workspaceStore, (state) => state.runtime.error);

  if (!error) return null;

  return (
    <div className="toast toast-end toast-bottom z-50 pointer-events-none">
      <div role="alert" className="alert alert-error alert-soft pointer-events-auto max-w-sm shadow-lg">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="size-5 shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span className="text-sm">{error}</span>
        <button
          type="button"
          className="btn btn-ghost btn-xs"
          aria-label={t`Dismiss error`}
          onClick={() => workspaceActions.setError(null)}
        >
          ✕
        </button>
      </div>
    </div>
  );
}
