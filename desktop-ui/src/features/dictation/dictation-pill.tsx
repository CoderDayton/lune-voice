import { backendClient } from "../../lib/backend/client";
import { AudioDots } from "./audio-dots";
import { MicIcon, PulseIndicator, StopIcon, XIcon } from "./pill-icons";
import { usePillState } from "./use-pill-state";
import { useLingui } from "@lingui/react/macro";

function pillAction(fn: () => Promise<unknown>, label: string) {
  fn().catch((e) => console.error(`[pill] ${label} failed`, e));
}

export function DictationPill() {
  const { t } = useLingui();
  const { mode, visible, recordingMode, audioLevel } = usePillState();

  if (!visible) return null;

  if (mode === "idle") {
    return (
      <div className="pill-shell">
        <button
          type="button"
          className="pill-idle"
          onClick={() => pillAction(backendClient.toggleDictation, "toggle_dictation")}
          aria-label={t`Start dictation`}
        >
          <MicIcon />
        </button>
      </div>
    );
  }

  const isToggle = recordingMode === "Toggle";

  return (
    <div className="pill-shell">
      <div className={`pill-active ${mode === "processing" ? "pill-processing" : ""}`} role="status" aria-live="polite" aria-label={mode === "processing" ? t`Processing` : t`Listening`}>
        {isToggle && (
          <button
            type="button"
            className="pill-btn pill-btn-cancel"
            onClick={() => pillAction(backendClient.cancelDictation, "cancel_dictation")}
            aria-label={t`Cancel dictation`}
          >
            <XIcon />
          </button>
        )}

        <div className="pill-center">
          {mode === "processing" ? <PulseIndicator label={t`Processing`} /> : <AudioDots level={audioLevel} />}
        </div>

        {isToggle && (
          <button
            type="button"
            className="pill-btn pill-btn-stop"
            onClick={() => pillAction(backendClient.stopDictation, "stop_dictation")}
            aria-label={t`Stop dictation`}
          >
            <StopIcon />
          </button>
        )}
      </div>
    </div>
  );
}
