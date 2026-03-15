import { invoke } from "@tauri-apps/api/core";
import { AudioDots } from "./audio-dots";
import { MicIcon, PulseIndicator, StopIcon, XIcon } from "./pill-icons";
import { usePillState } from "./use-pill-state";

export function DictationPill() {
  const { mode, visible, recordingMode } = usePillState();

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
        aria-label="Start dictation"
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
          aria-label="Cancel dictation"
        >
          <XIcon />
        </button>
      )}

      <div className="pill-center">
        {mode === "processing" ? <PulseIndicator /> : <AudioDots />}
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
          aria-label="Stop dictation"
        >
          <StopIcon />
        </button>
      )}
    </div>
  );
}
