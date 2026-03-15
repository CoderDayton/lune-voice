import { invoke } from "@tauri-apps/api/core";
import { AudioDots } from "./audio-dots";
import { MicIcon, PulseIndicator, StopIcon, XIcon } from "./pill-icons";
import { usePillState } from "./use-pill-state";
import { useLingui } from "@lingui/react/macro";

export function DictationPill() {
  const { t } = useLingui();
  const { mode, visible, recordingMode, audioLevel } = usePillState();

  if (!visible) return null;

  if (mode === "idle") {
    return (
      <button
        type="button"
        className="pill-idle"
        onClick={() =>
          invoke("toggle_dictation").catch((e) =>
            console.error("[pill] toggle_dictation failed", e),
          )
        }
        aria-label={t`Start dictation`}
      >
        <MicIcon />
      </button>
    );
  }

  const isToggle = recordingMode === "Toggle";

  return (
    <div className={`pill-active ${mode === "processing" ? "pill-processing" : ""}`}>
      {isToggle && (
        <button
          type="button"
          className="pill-btn pill-btn-cancel"
          onClick={() =>
            invoke("cancel_dictation").catch((e) =>
              console.error("[pill] cancel_dictation failed", e),
            )
          }
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
          onClick={() =>
            invoke("stop_dictation").catch((e) =>
              console.error("[pill] stop_dictation failed", e),
            )
          }
          aria-label={t`Stop dictation`}
        >
          <StopIcon />
        </button>
      )}
    </div>
  );
}
