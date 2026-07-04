import { listen } from '@tauri-apps/api/event';
import { useEffect, useEffectEvent, useRef, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import { LevelSmoother } from './level-smoother';

export type PillMode = 'idle' | 'active' | 'processing';

function modeFromStatus(status: string): PillMode {
  if (status === 'listening') return 'active';
  if (status === 'processing') return 'processing';
  return 'idle';
}

interface PillSettings {
  activationMode: string;
  pillVisibility: string;
}

/**
 * Pill *content* state. Whether the pill window is on screen is decided
 * entirely on the Rust side (runtime::publish_status → pill::show/hide_direct,
 * from the live service status + the pill_visibility setting) — this is the
 * single source of truth for visibility, so there is no frontend show/hide or
 * `data-visible` opacity gate that could desync from the mapped window.
 *
 * This hook only tracks what to render inside the pill: the mode
 * (idle/active/processing), the recording mode (Toggle vs Hold, for the
 * buttons), and the smoothed audio level for the dots.
 */
export function usePillState() {
  const [mode, setMode] = useState<PillMode>('idle');
  const [recordingMode, setRecordingMode] = useState<'Toggle' | 'Hold'>('Toggle');
  const [audioLevel, setAudioLevel] = useState(0);
  const smootherRef = useRef<LevelSmoother | null>(null);
  smootherRef.current ??= new LevelSmoother(setAudioLevel);

  // On mount: seed mode + recording mode from the current snapshot.
  useEffect(() => {
    let cancelled = false;
    backendClient.loadPillSnapshot().then(
      (snapshot) => {
        if (cancelled) return;
        setRecordingMode(snapshot.activationMode);
        setMode(modeFromStatus(snapshot.serviceStatus));
      },
      (e) => {
        if (!cancelled) console.warn('[pill] snapshot load failed', e);
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  // Stable event handlers via useEffectEvent — no stale closure risk.
  const onPillState = useEffectEvent((status: string) => {
    setMode(modeFromStatus(status));
  });

  const onPillSettings = useEffectEvent((s: PillSettings) => {
    if (s.activationMode === 'Toggle' || s.activationMode === 'Hold') {
      setRecordingMode(s.activationMode);
    }
  });

  // Smoothing + coalescing to at most one state update per animation frame
  // lives in LevelSmoother (see its docs — this is what kept the Linux pill
  // from stuttering).
  const onAudioLevel = useEffectEvent((level: number) => {
    smootherRef.current?.push(level);
  });

  // Listen to Rust backend events — consolidated into one effect.
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

  // Reset the audio level when the session ends (mode back to idle).
  useEffect(() => {
    if (mode === 'idle') {
      smootherRef.current?.reset();
      setAudioLevel(0);
    }
  }, [mode]);

  // Cancel a pending level frame on unmount.
  useEffect(() => () => smootherRef.current?.cancelPending(), []);

  return { mode, recordingMode, audioLevel };
}
