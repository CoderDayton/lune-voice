import { useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize } from "@tauri-apps/api/dpi";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AudioDots } from "./audio-dots";

type PillMode = "idle" | "active" | "processing";
type PillVis = "Always" | "WhenDictating" | "Never";

const PILL_IDLE_W = 48;
const PILL_IDLE_H = 48;
const PILL_ACTIVE_W = 260;
const PILL_ACTIVE_H = 56;

const AUTH_STORAGE_KEY = "lune.desktop.auth";
const READY_STATUSES = new Set(["ready", "listening", "processing"]);

function isAuthenticated(): boolean {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    return Boolean(parsed?.email && parsed?.name);
  } catch {
    return false;
  }
}

function modeFromStatus(status: string): PillMode {
  if (status === "listening") return "active";
  if (status === "processing") return "processing";
  return "idle";
}

function shouldShow(vis: PillVis, mode: PillMode, serviceReady: boolean): boolean {
  if (vis === "Never" || !serviceReady) return false;
  if (vis === "WhenDictating") return mode !== "idle";
  return true; // Always
}

interface PillSettings {
  activationMode?: string;
  pillVisibility?: string;
}

export function DictationPill() {
  const [mode, setMode] = useState<PillMode>("idle");
  const [pillVis, setPillVis] = useState<PillVis>("Always");
  const [serviceReady, setServiceReady] = useState(false);
  const [recordingMode, setRecordingMode] = useState<"Toggle" | "Hold">("Toggle");
  const shownRef = useRef(false);

  // On mount: check auth + load snapshot for initial readiness
  useEffect(() => {
    if (!isAuthenticated()) return;

    invoke<{ serviceStatus: string; settings: { activationMode: string; pillVisibility: string } }>(
      "load_workspace_snapshot",
    ).then(
      (snapshot) => {
        const am = snapshot.settings.activationMode;
        if (am === "Hold" || am === "Toggle") setRecordingMode(am);
        const pv = snapshot.settings.pillVisibility;
        if (pv === "Always" || pv === "WhenDictating" || pv === "Never") setPillVis(pv);
        if (READY_STATUSES.has(snapshot.serviceStatus)) {
          setServiceReady(true);
          setMode(modeFromStatus(snapshot.serviceStatus));
        }
      },
      () => {},
    );
  }, []);

  // Listen to lune://pill-state events from Rust backend
  useEffect(() => {
    const unlistenState = listen<string>("lune://pill-state", (event) => {
      if (!isAuthenticated()) {
        setServiceReady(false);
        return;
      }
      if (READY_STATUSES.has(event.payload)) {
        setServiceReady(true);
        setMode(modeFromStatus(event.payload));
      } else {
        setServiceReady(false);
        setMode("idle");
      }
    });

    // Track settings changes from the main window
    const unlistenSettings = listen<PillSettings>("lune://pill-settings", (event) => {
      const s = event.payload;
      if (s.activationMode === "Toggle" || s.activationMode === "Hold") {
        setRecordingMode(s.activationMode);
      }
      if (s.pillVisibility === "Always" || s.pillVisibility === "WhenDictating" || s.pillVisibility === "Never") {
        setPillVis(s.pillVisibility);
      }
    });

    return () => {
      unlistenState.then((fn) => fn());
      unlistenSettings.then((fn) => fn());
    };
  }, []);

  const visible = shouldShow(pillVis, mode, serviceReady);

  // Show/hide the Tauri window based on visibility
  useEffect(() => {
    if (visible && !shownRef.current) {
      shownRef.current = true;
      void invoke("show_pill_window");
    } else if (!visible && shownRef.current) {
      shownRef.current = false;
      void invoke("hide_pill_window");
    }
  }, [visible]);

  // Resize + reposition when mode changes (only while visible)
  useEffect(() => {
    if (!visible) return;
    const win = getCurrentWindow();
    const w = mode === "idle" ? PILL_IDLE_W : PILL_ACTIVE_W;
    const h = mode === "idle" ? PILL_IDLE_H : PILL_ACTIVE_H;
    win.setSize(new LogicalSize(w, h)).then(() => {
      void invoke("reposition_pill_center");
    });
  }, [mode, visible]);

  if (!visible) return null;

  const handleIdleClick = () => {
    void invoke("toggle_dictation");
  };

  const handleStop = () => {
    void invoke("stop_dictation");
  };

  const handleCancel = () => {
    void invoke("cancel_dictation");
  };

  if (mode === "idle") {
    return (
      <button
        type="button"
        className="pill-idle"
        onClick={handleIdleClick}
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
          onClick={handleCancel}
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
          onClick={handleStop}
          aria-label="Stop dictation"
        >
          <StopIcon />
        </button>
      )}
    </div>
  );
}

function MicIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="size-5">
      <path d="M8.25 4.5a3.75 3.75 0 1 1 7.5 0v8.25a3.75 3.75 0 1 1-7.5 0V4.5Z" />
      <path d="M6 10.5a.75.75 0 0 1 .75.75v1.5a5.25 5.25 0 1 0 10.5 0v-1.5a.75.75 0 0 1 1.5 0v1.5a6.751 6.751 0 0 1-6 6.709v2.291h3a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1 0-1.5h3v-2.291a6.751 6.751 0 0 1-6-6.709v-1.5A.75.75 0 0 1 6 10.5Z" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="size-4">
      <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="size-4">
      <rect x="5" y="5" width="10" height="10" rx="1" />
    </svg>
  );
}

function PulseIndicator() {
  return <span className="pill-pulse" aria-label="Processing" />;
}
