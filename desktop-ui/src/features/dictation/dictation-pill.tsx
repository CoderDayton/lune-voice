import { useCallback, useRef } from "react";
import { backendClient } from "../../lib/backend/client";
import { AudioDots } from "./audio-dots";
import { MicIcon, PulseIndicator, StopIcon, XIcon } from "./pill-icons";
import { usePillState } from "./use-pill-state";
import { useLingui } from "@lingui/react/macro";

const PILL_DEBOUNCE_MS = 200;

function fire(fn: () => Promise<unknown>, tag: string) {
  fn().catch((e) => console.error(`[pill] ${tag} failed`, e));
}

export function DictationPill() {
  const { t } = useLingui();
  const { mode, visible, recordingMode, audioLevel } = usePillState();
  const cancelRef = useRef(false);
  const lastFireRef = useRef(0);

  const debouncedFire = useCallback((fn: () => Promise<unknown>, tag: string) => {
    const now = Date.now();
    if (now - lastFireRef.current < PILL_DEBOUNCE_MS) return;
    lastFireRef.current = now;
    fire(fn, tag);
  }, []);

  const isToggle = recordingMode === "Toggle";
  const isIdle = mode === "idle";
  const isProcessing = mode === "processing";

  const handleCancel = () => {
    cancelRef.current = true;
    debouncedFire(() => backendClient.cancelDictation(), "cancel");
    setTimeout(() => { cancelRef.current = false; }, 300);
  };

  const rootClass = `pill-root${cancelRef.current ? " pill-cancelled" : ""}`;

  return (
    <div className={rootClass} data-visible={visible || undefined}>
      <div className="pill-shell">
        {isIdle ? (
          <button
            type="button"
            className="pill-idle"
            onClick={() => debouncedFire(() => backendClient.toggleDictation(), "toggle")}
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
                onClick={handleCancel}
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
                onClick={() => debouncedFire(() => backendClient.stopDictation(), "stop")}
                aria-label={t`Stop dictation`}
              >
                <StopIcon />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
