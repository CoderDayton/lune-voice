import { backendClient } from "../../lib/backend/client";
import { AudioDots } from "./audio-dots";
import { MicIcon, PulseIndicator, StopIcon, XIcon } from "./pill-icons";
import { usePillState } from "./use-pill-state";
import { useLingui } from "@lingui/react/macro";

function fire(fn: () => Promise<unknown>, tag: string) {
  fn().catch((e) => console.error(`[pill] ${tag} failed`, e));
}

export function DictationPill() {
  const { t } = useLingui();
  const { mode, visible, recordingMode, audioLevel } = usePillState();

  const isToggle = recordingMode === "Toggle";
  const isIdle = mode === "idle";
  const isProcessing = mode === "processing";

  // TODO: pill temporarily disabled while fixing Wayland compositor ghost artifacts
  return (
    <div className="pill-root" data-visible={visible || undefined}>
      {false && visible && (
        <div className="pill-shell">
          {isIdle ? (
            <button
              type="button"
              className="pill-idle"
              onClick={() => fire(backendClient.toggleDictation, "toggle")}
              aria-label={t`Start dictation`}
            >
              <MicIcon />
            </button>
          ) : (
            <div
              className={`pill-active${isProcessing ? " pill-processing" : ""}`}
              role="status"
              aria-live="polite"
              aria-label={isProcessing ? t`Processing` : t`Listening`}
            >
              {isToggle && (
                <button
                  type="button"
                  className="pill-btn pill-btn-cancel"
                  onClick={() => fire(backendClient.cancelDictation, "cancel")}
                  aria-label={t`Cancel dictation`}
                >
                  <XIcon />
                </button>
              )}

              <div className="pill-center">
                {isProcessing ? (
                  <PulseIndicator label={t`Processing`} />
                ) : (
                  <AudioDots level={audioLevel} />
                )}
              </div>

              {isToggle && (
                <button
                  type="button"
                  className="pill-btn pill-btn-stop"
                  onClick={() => fire(backendClient.stopDictation, "stop")}
                  aria-label={t`Stop dictation`}
                >
                  <StopIcon />
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
