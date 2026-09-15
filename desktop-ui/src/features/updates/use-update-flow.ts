// SPDX-License-Identifier: GPL-3.0-or-later

import { listen } from '@tauri-apps/api/event';
import { useCallback, useEffect, useState } from 'react';
import { backendClient } from '../../lib/backend/client';
import { isTauriRuntime } from '../../lib/backend/runtime';
import type { UpdateStateDto } from '../../lib/backend/types';

/** Backend event carrying every transition of the update flow. */
const UPDATE_EVENT = 'lune://update';

/** Phases where there is nothing for the user to answer, so no modal. */
const IDLE_PHASES = new Set(['idle']);

export interface UpdateFlow {
  state: UpdateStateDto | null;
  /** Whether the modal should be showing. */
  open: boolean;
  /** Accept the offer: download, install, restart. */
  install: () => void;
  /** Close the modal and forget the offer until the next check. */
  dismiss: () => void;
  /** Run the check again after a failure. */
  retry: () => void;
}

/**
 * Subscribes to the backend's update flow and decides when the modal is up.
 *
 * The mount-time read is not belt-and-braces: a launch check runs twenty
 * seconds after startup and can find an update while the main window is closed
 * (the app starts in the tray, or the window was closed to it). The event
 * announcing that has no window to reach, so the backend keeps the state and
 * this picks it up the first time a window exists.
 */
export function useUpdateFlow(): UpdateFlow {
  const [state, setState] = useState<UpdateStateDto | null>(null);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    let cancelled = false;

    void backendClient
      .updateState()
      .then((current) => {
        // Never clobber a live event with the mount-time read: the fetch is
        // async, and a check that started in between has the newer answer.
        if (!cancelled) setState((prev) => prev ?? current);
      })
      .catch((err) => {
        // Nothing to show and nothing the user can do about it — the modal
        // simply stays closed rather than reporting a failure to read a status.
        console.warn('[update-flow] could not read the update state', err);
      });

    const unlisten = listen<UpdateStateDto>(UPDATE_EVENT, (event) => {
      setState(event.payload);
    });

    return () => {
      cancelled = true;
      void unlisten.then((fn) => fn());
    };
  }, []);

  const install = useCallback(() => {
    // The backend answers by publishing `downloading` → `installing`, so the
    // modal follows the real progress rather than a local guess at it. A
    // rejection here is reported the same way, as an `install-failed` state.
    void backendClient.installPendingUpdate().catch((err) => {
      console.error('[update-flow] install failed to start', err);
    });
  }, []);

  const dismiss = useCallback(() => {
    // Close immediately rather than waiting for the round trip: the backend
    // has nothing to say about a dismissal, and a modal that lingers until IPC
    // answers reads as an unresponsive button.
    setState((prev) => (prev ? { ...prev, phase: 'idle' } : prev));
    void backendClient.dismissUpdate().catch((err) => {
      console.warn('[update-flow] dismiss failed', err);
    });
  }, []);

  const retry = useCallback(() => {
    void backendClient.startUpdateCheck().catch((err) => {
      console.error('[update-flow] retry failed to start', err);
    });
  }, []);

  return {
    state,
    open: state != null && !IDLE_PHASES.has(state.phase),
    install,
    dismiss,
    retry,
  };
}
