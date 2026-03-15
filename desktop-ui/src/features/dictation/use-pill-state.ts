import { useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize } from "@tauri-apps/api/dpi";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { attachSessionStream } from "../../lib/backend/stream";
import type { SessionStreamEvent } from "../../lib/backend/types";

export type PillMode = "idle" | "active" | "processing";
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
  return true;
}

interface PillSettings {
  activationMode: string;
  pillVisibility: string;
}

export function usePillState() {
  const [mode, setMode] = useState<PillMode>("idle");
  const [pillVis, setPillVis] = useState<PillVis>("Always");
  const [serviceReady, setServiceReady] = useState(false);
  const [recordingMode, setRecordingMode] = useState<"Toggle" | "Hold">("Toggle");
  const [audioLevel, setAudioLevel] = useState(0);
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
      (e) => console.warn("[pill] snapshot load failed, staying hidden", e),
    );
  }, []);

  // Listen to Rust backend events
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
      unlistenState.then((fn) => fn()).catch(() => {});
      unlistenSettings.then((fn) => fn()).catch(() => {});
    };
  }, []);

  // Subscribe to session stream for audio levels
  useEffect(() => {
    let detach = () => {};
    attachSessionStream((event: SessionStreamEvent) => {
      if (event.type === "audioLevel") {
        setAudioLevel(event.level);
      } else if (event.type === "sessionStopped" || event.type === "sessionError") {
        setAudioLevel(0);
      }
    }).then(
      (fn) => { detach = fn; },
      () => {},
    );
    return () => detach();
  }, []);

  const visible = shouldShow(pillVis, mode, serviceReady);

  // Show/hide the Tauri window based on visibility.
  // Reset shownRef on mount so HMR/remount re-syncs with Tauri window state.
  useEffect(() => {
    shownRef.current = false;
  }, []);

  useEffect(() => {
    if (visible && !shownRef.current) {
      shownRef.current = true;
      invoke("show_pill_window").catch((e) =>
        console.warn("[pill] show_pill_window failed", e),
      );
    } else if (!visible && shownRef.current) {
      shownRef.current = false;
      invoke("hide_pill_window").catch((e) =>
        console.warn("[pill] hide_pill_window failed", e),
      );
    }
  }, [visible]);

  // Resize + reposition when mode changes (only while visible)
  useEffect(() => {
    if (!visible) return;
    const win = getCurrentWindow();
    const w = mode === "idle" ? PILL_IDLE_W : PILL_ACTIVE_W;
    const h = mode === "idle" ? PILL_IDLE_H : PILL_ACTIVE_H;
    win.setSize(new LogicalSize(w, h)).then(
      () => invoke("reposition_pill_center").catch(() => {}),
      (e) => console.warn("[pill] setSize failed", e),
    );
  }, [mode, visible]);

  return { mode, visible, recordingMode, audioLevel };
}
