import { invoke } from '@tauri-apps/api/core';
import { LogicalSize } from '@tauri-apps/api/dpi';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { useEffect, useEffectEvent, useRef, useState } from 'react';
import type { PillVisibilityValue } from '../../features/settings/data/settings-data';
import { backendClient } from '../../lib/backend/client';

export type PillMode = 'idle' | 'active' | 'processing';
type PillVis = PillVisibilityValue;

// Fixed window size for all modes — avoids compositor ghost artifacts
// on transparent Wayland windows during resize (wry #1667, tauri #14924).
const PILL_W = 260;
const PILL_H = 56;

const READY_STATUSES = new Set(['ready', 'listening', 'processing']);

function modeFromStatus(status: string): PillMode {
  if (status === 'listening') return 'active';
  if (status === 'processing') return 'processing';
  return 'idle';
}

function shouldShow(vis: PillVis, mode: PillMode, serviceReady: boolean): boolean {
  if (vis === 'Never' || !serviceReady) return false;
  if (vis === 'WhenDictating') return mode !== 'idle';
  return true;
}

interface PillSettings {
  activationMode: string;
  pillVisibility: string;
}

const PILL_VIS_VALUES = new Set<string>(['Always', 'WhenDictating', 'Never']);

export function usePillState() {
  const [mode, setMode] = useState<PillMode>('idle');
  const [pillVis, setPillVis] = useState<PillVis>('Always');
  const [serviceReady, setServiceReady] = useState(false);
  const [recordingMode, setRecordingMode] = useState<'Toggle' | 'Hold'>('Toggle');
  const [audioLevel, setAudioLevel] = useState(0);
  const smoothedRef = useRef(0);
  const shownRef = useRef(false);

  // On mount: load workspace snapshot for initial readiness
  useEffect(() => {
    let cancelled = false;
    backendClient.loadPillSnapshot().then(
      (snapshot) => {
        if (cancelled) return;
        setRecordingMode(snapshot.activationMode);
        setPillVis(snapshot.pillVisibility);
        if (READY_STATUSES.has(snapshot.serviceStatus)) {
          setServiceReady(true);
          setMode(modeFromStatus(snapshot.serviceStatus));
        }
      },
      (e) => {
        if (!cancelled) console.warn('[pill] snapshot load failed, staying hidden', e);
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  // Stable event handlers via useEffectEvent — no stale closure risk
  const onPillState = useEffectEvent((status: string) => {
    if (READY_STATUSES.has(status)) {
      setServiceReady(true);
      setMode(modeFromStatus(status));
    } else {
      setServiceReady(false);
      setMode('idle');
    }
  });

  const onPillSettings = useEffectEvent((s: PillSettings) => {
    if (s.activationMode === 'Toggle' || s.activationMode === 'Hold') {
      setRecordingMode(s.activationMode);
    }
    if (PILL_VIS_VALUES.has(s.pillVisibility)) {
      setPillVis(s.pillVisibility as PillVis);
    }
  });

  const onAudioLevel = useEffectEvent((level: number) => {
    const alpha = level > smoothedRef.current ? 0.35 : 0.15;
    smoothedRef.current += alpha * (level - smoothedRef.current);
    setAudioLevel(smoothedRef.current);
  });

  // Listen to Rust backend events — consolidated into one effect with stable handlers
  useEffect(() => {
    const unlistenState = listen<string>('lune://pill-state', (e) => onPillState(e.payload));
    const unlistenSettings = listen<PillSettings>('lune://pill-settings', (e) =>
      onPillSettings(e.payload),
    );
    const unlistenLevel = listen<number>('lune://audio-level', (e) => onAudioLevel(e.payload));

    return () => {
      unlistenState.then((fn) => fn()).catch(() => {});
      unlistenSettings.then((fn) => fn()).catch(() => {});
      unlistenLevel.then((fn) => fn()).catch(() => {});
    };
  }, []);

  // Reset audio level when session ends (tracked via pill-state event)
  useEffect(() => {
    if (mode === 'idle') {
      setAudioLevel(0);
      smoothedRef.current = 0;
    }
  }, [mode]);

  const visible = shouldShow(pillVis, mode, serviceReady);

  // Show/hide the Tauri window based on visibility.
  // Reset shownRef on mount so HMR/remount re-syncs with Tauri window state.
  useEffect(() => {
    shownRef.current = false;
  }, []);

  useEffect(() => {
    if (visible && !shownRef.current) {
      shownRef.current = true;
      invoke('show_pill_window').catch((e) => console.warn('[pill] show_pill_window failed', e));
    } else if (!visible && shownRef.current) {
      shownRef.current = false;
      invoke('hide_pill_window').catch((e) => console.warn('[pill] hide_pill_window failed', e));
    }
  }, [visible]);

  // Set fixed window size on first show and reposition
  useEffect(() => {
    if (!visible) return;
    const win = getCurrentWindow();
    win.setSize(new LogicalSize(PILL_W, PILL_H)).then(
      () => invoke('reposition_pill_center').catch(() => {}),
      (e) => console.warn('[pill] setSize failed', e),
    );
  }, [visible]);

  // Pill show/hide is now driven directly from Rust via publish_status →
  // show_pill_direct/hide_pill_direct, so no frontend hide/show cycle needed.

  return { mode, visible, recordingMode, audioLevel };
}
