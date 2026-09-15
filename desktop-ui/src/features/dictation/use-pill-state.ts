// SPDX-License-Identifier: GPL-3.0-or-later

import { listen } from '@tauri-apps/api/event';
import { useEffect, useEffectEvent, useRef, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import { LevelSmoother } from './level-smoother';
import { ModeGate } from './mode-gate';

export type PillMode = 'idle' | 'active' | 'processing';

function modeFromStatus(status: string): PillMode {
  if (status === 'listening') return 'active';
  if (status === 'processing') return 'processing';
  return 'idle';
}

interface PillSettings {
  pillLocked?: boolean;
}

/**
 * Pill *content* state. Whether the pill window is on screen is decided
 * entirely on the Rust side (runtime::publish_status → pill::show/hide_direct,
 * from the live service status + the pill_visibility setting) — this is the
 * single source of truth for visibility, so there is no frontend show/hide or
 * `data-visible` opacity gate that could desync from the mapped window.
 *
 * This hook only tracks what to render inside the pill: the mode
 * (idle/active/processing), whether dragging is locked, and the smoothed audio
 * level for the dots.
 */
export function usePillState() {
  const [mode, setMode] = useState<PillMode>('idle');
  const [locked, setLocked] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const smootherRef = useRef<LevelSmoother | null>(null);
  smootherRef.current ??= new LevelSmoother(setAudioLevel);
  // Which mode is *rendered* — see ModeGate for why that isn't always the one
  // the backend last published.
  const gateRef = useRef<ModeGate | null>(null);
  gateRef.current ??= new ModeGate(setMode);

  // On mount: seed mode + lock state from the current snapshot.
  useEffect(() => {
    let cancelled = false;
    backendClient.loadPillSnapshot().then(
      (snapshot) => {
        if (cancelled) return;
        setLocked(snapshot.pillLocked);
        gateRef.current?.push(modeFromStatus(snapshot.serviceStatus));
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
    gateRef.current?.push(modeFromStatus(status));
  });

  const onPillSettings = useEffectEvent((s: PillSettings) => {
    if (typeof s.pillLocked === 'boolean') {
      setLocked(s.pillLocked);
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

  // Cancel pending work on unmount.
  useEffect(
    () => () => {
      smootherRef.current?.cancelPending();
      gateRef.current?.cancelPending();
    },
    [],
  );

  return { mode, audioLevel, locked };
}
