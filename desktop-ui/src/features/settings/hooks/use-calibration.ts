import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import type {
  CalibrationDiagnostics,
  CalibrationHandle,
  CalibrationPhase,
  CalibrationStreamEvent,
} from '../../../lib/backend/types';
import { settingsToSaveInput } from '../../../store/workspace-mappers';
import { workspaceActions, workspaceStore } from '../../../store/workspace-store';

export interface CalibrationRecommendation {
  speechThreshold: number;
  speechThresholdEnd: number;
  diagnostics: CalibrationDiagnostics;
}

export interface UseCalibrationApi {
  phase: CalibrationPhase;
  vadScore: number;
  energy: number;
  recommendation: CalibrationRecommendation | null;
  errorMessage: string | null;
  start: () => Promise<void>;
  advance: () => Promise<void>;
  cancel: () => Promise<void>;
  apply: () => Promise<void>;
  reset: () => void;
}

export function useCalibration(): UseCalibrationApi {
  const [phase, setPhase] = useState<CalibrationPhase>('idle');
  const [vadScore, setVadScore] = useState(0);
  const [energy, setEnergy] = useState(0);
  const [recommendation, setRecommendation] = useState<CalibrationRecommendation | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const handleRef = useRef<CalibrationHandle | null>(null);

  const onEvent = useCallback((event: CalibrationStreamEvent) => {
    switch (event.type) {
      case 'phaseChanged':
        setPhase(event.phase);
        break;
      case 'frame':
        setVadScore(event.vadScore);
        setEnergy(event.energy);
        break;
      case 'result':
        setRecommendation({
          speechThreshold: event.speechThreshold,
          speechThresholdEnd: event.speechThresholdEnd,
          diagnostics: event.diagnostics,
        });
        break;
      case 'error':
        setErrorMessage(event.message);
        setPhase('failed');
        break;
    }
  }, []);

  const reset = useCallback(() => {
    setPhase('idle');
    setVadScore(0);
    setEnergy(0);
    setRecommendation(null);
    setErrorMessage(null);
  }, []);

  const start = useCallback(async () => {
    if (handleRef.current) {
      await handleRef.current.cancel().catch(() => {});
      handleRef.current = null;
    }
    reset();
    setPhase('noise');
    try {
      handleRef.current = await backendClient.calibrateStart(onEvent);
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : String(e));
      setPhase('failed');
    }
  }, [onEvent, reset]);

  const advance = useCallback(async () => {
    if (!handleRef.current) return;
    try {
      await handleRef.current.advance();
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : String(e));
      setPhase('failed');
    }
  }, []);

  const cancel = useCallback(async () => {
    const handle = handleRef.current;
    handleRef.current = null;
    if (handle) {
      await handle.cancel().catch(() => {});
    }
    reset();
  }, [reset]);

  const apply = useCallback(async () => {
    if (!recommendation) return;
    workspaceActions.updateSetting('speechThreshold', recommendation.speechThreshold.toFixed(2));
    workspaceActions.updateSetting(
      'speechThresholdEnd',
      recommendation.speechThresholdEnd.toFixed(2),
    );
    try {
      const snapshot = await backendClient.saveWorkspaceSettings(
        settingsToSaveInput(workspaceStore.state.settings),
      );
      workspaceActions.hydrate(snapshot);
    } catch (e) {
      const snapshot = await backendClient.loadWorkspaceSnapshot();
      workspaceActions.hydrate(snapshot);
      setErrorMessage(e instanceof Error ? e.message : String(e));
    }
  }, [recommendation]);

  useEffect(() => {
    return () => {
      const handle = handleRef.current;
      handleRef.current = null;
      if (handle) handle.cancel().catch(() => {});
    };
  }, []);

  return useMemo(
    () => ({
      phase,
      vadScore,
      energy,
      recommendation,
      errorMessage,
      start,
      advance,
      cancel,
      apply,
      reset,
    }),
    [phase, vadScore, energy, recommendation, errorMessage, start, advance, cancel, apply, reset],
  );
}
